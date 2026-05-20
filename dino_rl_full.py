import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import argparse

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--play", action="store_true")
args = parser.parse_args()

# ---------- CONFIG ----------
WIDTH, HEIGHT = 900, 300
GROUND_Y = 240
FPS = 60

MODEL_PATH = "dino_model.pth"
CKPT_PATH = "dino_ckpt.pth"

# ---------- NETWORK ----------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.net(x)

# ---------- AGENT ----------
class Agent:
    def __init__(self, eval_mode=False):
        self.model = Net()
        self.target = Net()
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)
        self.mem = deque(maxlen=50000)

        self.eps = 1.0
        self.episode = 0
        self.eval_mode = eval_mode

        self.load()
        self.target.load_state_dict(self.model.state_dict())

        if self.eval_mode:
            self.eps = 0

    def act(self, s):
        if not self.eval_mode and random.random() < self.eps:
            return random.randint(0,1)
        with torch.no_grad():
            return int(torch.argmax(self.model(torch.tensor(s, dtype=torch.float32))))

    def train(self):
        if self.eval_mode or len(self.mem) < 64:
            return

        batch = random.sample(self.mem, 64)
        s,a,r,ns,d = zip(*batch)

        s = torch.tensor(s, dtype=torch.float32)
        ns = torch.tensor(ns, dtype=torch.float32)
        a = torch.tensor(a).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32)

        q = self.model(s).gather(1,a).squeeze()
        with torch.no_grad():
            nq = self.target(ns).max(1)[0]
            target = r + (1-d)*0.99*nq

        loss = nn.MSELoss()(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.eps > 0.05:
            self.eps *= 0.995

    # ---------- SAVE ----------
    def save(self, final=False):
        torch.save(self.model.state_dict(), MODEL_PATH)
        torch.save({
            "episode": self.episode,
            "eps": self.eps
        }, CKPT_PATH)

        if final:
            print(f"💾 Final model saved @ episode {self.episode}")
        else:
            print(f"💾 Saved @ episode {self.episode}")

    # ---------- LOAD ----------
    def load(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(CKPT_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH))
            ck = torch.load(CKPT_PATH)
            self.episode = ck.get("episode", 0)
            self.eps = ck.get("eps", 1.0)

            print("✅ Model loaded")
            print(f"Episode = {self.episode}")
        else:
            print("⚠ Training from scratch")

# ---------- GAME ----------
class Game:
    def __init__(self):
        self.reset()

    def reset(self):
        self.y = GROUND_Y
        self.vel = 0
        self.jump = False

        self.obstacles = []
        self.score = 0

        self.speed = 6
        self.spawn_gap = random.randint(250, 400)
        self.distance_since_last = 0

        return self.state()

    def state(self):
        if self.obstacles:
            ob = self.obstacles[0]
            dist = ob[0] - 100
            height = ob[3]
        else:
            dist, height = WIDTH, 0
        return np.array([dist/900, height/100, self.vel/10, int(self.jump)])

    def step(self, action):
        reward = 1
        done = False

        if action == 1 and not self.jump:
            self.vel = -15
            self.jump = True

        self.vel += 1
        self.y += self.vel

        if self.y >= GROUND_Y:
            self.y = GROUND_Y
            self.jump = False

        # difficulty scaling
        self.score += 1
        self.speed = min(12, 6 + self.score // 200)

        min_gap = max(120, 300 - self.score // 5)
        max_gap = max(200, 450 - self.score // 5)

        self.distance_since_last += self.speed

        if self.distance_since_last > self.spawn_gap:
            height = random.choice([30, 40, 50])
            self.obstacles.append([WIDTH, GROUND_Y-height, 20, height])

            self.spawn_gap = random.randint(min_gap, max_gap)
            self.distance_since_last = 0

        for ob in self.obstacles:
            ob[0] -= self.speed

        self.obstacles = [o for o in self.obstacles if o[0] > -20]

        dino = pygame.Rect(100, self.y-40, 40, 40)
        for ob in self.obstacles:
            if dino.colliderect(pygame.Rect(*ob)):
                reward = -100
                done = True

        return self.state(), reward, done

# ---------- GUI ----------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 20)

agent = Agent(eval_mode=args.play)
env = Game()
state = env.reset()

stars = [(random.randint(0,WIDTH), random.randint(0,150)) for _ in range(70)]
clouds = [[random.randint(0,WIDTH), random.randint(20,120)] for _ in range(3)]

def draw_dino(x, y, frame):
    pygame.draw.rect(screen,(255,255,255),(x+5,y-30,25,20))
    pygame.draw.rect(screen,(255,255,255),(x+25,y-40,15,15))
    pygame.draw.rect(screen,(0,0,0),(x+35,y-35,3,3))
    pygame.draw.rect(screen,(255,255,255),(x,y-25,10,5))
    if frame:
        pygame.draw.rect(screen,(255,255,255),(x+10,y-10,5,10))
    else:
        pygame.draw.rect(screen,(255,255,255),(x+20,y-10,5,10))

frame = 0
timer = 0

running = True
while running:
    clock.tick(FPS)

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    action = agent.act(state)
    ns, r, done = env.step(action)

    agent.mem.append((state, action, r, ns, done))
    agent.train()
    state = ns

    if done:
        agent.episode += 1
        if not args.play:
            agent.save()
        state = env.reset()

    # DRAW
    screen.fill((0,0,0))

    for s in stars:
        pygame.draw.circle(screen,(255,255,255),s,1)

    pygame.draw.circle(screen,(255,255,255),(750,60),25)
    pygame.draw.circle(screen,(0,0,0),(760,60),20)

    for c in clouds:
        pygame.draw.line(screen,(255,255,255),(c[0],c[1]),(c[0]+60,c[1]),2)
        c[0] -= 1
        if c[0] < -60:
            c[0] = WIDTH

    pygame.draw.line(screen,(255,255,255),(0,GROUND_Y),(WIDTH,GROUND_Y),2)

    timer += 1
    if timer > 10:
        frame = 1-frame
        timer = 0

    draw_dino(100, env.y, frame)

    for ob in env.obstacles:
        pygame.draw.rect(screen,(255,255,255),ob)

    txt = font.render(
        f"Ep:{agent.episode} Score:{env.score} Speed:{env.speed} Eps:{agent.eps:.2f} Mode:{'PLAY' if args.play else 'TRAIN'}",
        True,(255,255,255)
    )
    screen.blit(txt,(10,10))

    pygame.display.flip()

pygame.quit()

# FINAL SAVE
if not args.play:
    agent.save(final=True)