import numpy as np
from tqdm import tqdm

class AxelrodModel:
    def __init__(self, L, F, q):
        self.L = L
        self.F = F
        self.q = q
        self.lattice = self.initialize_lattice()

    def initialize_lattice(self):
        return np.random.poisson(self.q, (self.L, self.L, self.F))

    def interact(self, i1, j1, i2, j2):
        features1 = self.lattice[i1, j1]
        features2 = self.lattice[i2, j2]

        common_features = np.where(features1 == features2)[0]
        different_features = np.where(features1 != features2)[0]

        if len(common_features) == 0 or len(different_features) == 0:
            return

        random_feature = np.random.choice(different_features)
        self.lattice[i1, j1, random_feature] = self.lattice[i2, j2, random_feature]

    def step(self):
        i1, j1 = np.random.randint(self.L, size=2)
        direction = np.random.choice(['up', 'down', 'left', 'right'])

        if direction == 'up':
            i2, j2 = (i1-1) % self.L, j1
        elif direction == 'down':
            i2, j2 = (i1+1) % self.L, j1
        elif direction == 'left':
            i2, j2 = i1, (j1-1) % self.L
        else:
            i2, j2 = i1, (j1+1) % self.L

        self.interact(i1, j1, i2, j2)

    def largest_domain(self):
        visited = np.zeros((self.L, self.L), dtype=bool)
        max_domain_size = 0

        for i in range(self.L):
            for j in range(self.L):
                if not visited[i, j]:
                    size, domain = self.domain_size(i, j, visited)
                    max_domain_size = max(max_domain_size, size)

        return max_domain_size

    def domain_size(self, i, j, visited):
        if visited[i, j]:
            return 0, []

        stack = [(i, j)]
        domain = [self.lattice[i, j].tolist()]
        visited[i, j] = True
        size = 1

        while stack:
            x, y = stack.pop()
            neighbors = [
                ((x-1) % self.L, y),
                ((x+1) % self.L, y),
                (x, (y-1) % self.L),
                (x, (y+1) % self.L)
            ]

            for nx, ny in neighbors:
                if not visited[nx, ny] and all(self.lattice[nx, ny] == self.lattice[i, j]):
                    visited[nx, ny] = True
                    stack.append((nx, ny))
                    domain.append(self.lattice[nx, ny].tolist())
                    size += 1

        return size, domain

def simulate(L, F, q, steps):
    model = AxelrodModel(L, F, q)
    for _ in tqdm(range(steps)):
        model.step()
    smax = model.largest_domain()
    return smax / (L * L)
