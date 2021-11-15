import simanneal
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
from simanneal.anneal import time_string
import sys


class NQueensBaseAnnealer(simanneal.Annealer):
    def __init__(self, n_queens=7, load_state=None):
        self.n_queens = n_queens
        super().__init__(initial_state=np.arange(n_queens),
                         load_state=load_state)
        self.reset_metrics()

    def reset_metrics(self):
        self.epochs = 0
        self.T_hist = []
        self.E_hist = []
        self.accept_hist = []
        self.improv_hist = []
        self.steps_hist = []

    def energy(self):
        n_collisions = 0
        # Calculate number of collisions
        for (i, j) in enumerate(self.state):
            for (m, n) in enumerate(self.state[i+1:]):
                if n == j:
                    n_collisions += 1
                elif abs(j - n) == m + 1:
                    n_collisions += 1
        if n_collisions == 0:
            self.user_exit = True
        return n_collisions

    def move(self):
        self.epochs += 1
        # swap places of queens
        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        self.state[a], self.state[b] = self.state[b], self.state[a]

    def anneal(self, updates=None):
        if updates:
            self.updates = updates
        self.reset_metrics()
        r = super().anneal()
        print()
        return r

    def plot_evolution(self):
        if len(self.T_hist) < 2:
            print('No hay suficientes datos para la representación')
            return
        # Plot temperature, energy, accept, improve
        fig = plt.figure(figsize=(22, 10), constrained_layout=True)
        gs = fig.add_gridspec(2, 2)
        fig.suptitle('Evolución del algoritmo')
        metrics = [(self.T_hist, 'Temperatura'),
                   (self.E_hist, 'Energía'),
                   (self.accept_hist, 'Aceptación'),
                   (self.improv_hist, 'Mejora')]
        for i in range(2):
            for j in range(2):
                atr = metrics[i * 2 + j]
                f_ax = fig.add_subplot(gs[i, j])
                f_ax.set_title(f'{atr[1]}(iteraciones)')
                if atr[1] == 'Energía':
                    f_ax.set_ylim(ymax=max(self.E_hist[1:]) + 3)
                    m_idx = np.argmin(self.E_hist)
                    if self.best_energy != 0:
                        f_ax.annotate(
                            text='Mínimo',
                            xy=(self.steps_hist[m_idx], self.E_hist[m_idx]),
                            arrowprops=dict(facecolor='red', shrink=0.05),
                        )
                    else:
                        f_ax.plot(self.steps_hist + [self.epochs],
                                  atr[0] + [0], color='#1f77b4')
                f_ax.plot(self.steps_hist, atr[0])
        plt.show()
        pass

    def default_update(self, step, T, E, acceptance, improvement):
        clear_output(wait=True)
        self.T_hist.append(T)
        self.E_hist.append(E)
        self.accept_hist.append(acceptance)
        self.improv_hist.append(improvement)
        self.steps_hist.append(step)

        elapsed = time.time() - self.start
        header = (' Temperature        Energy    Accept   Improve     Elapsed'
                  '   Remaining')
        if step == 0:
            print(header, file=sys.stdout)
            print('%12.5f  %12.2f                      %s            ' %
                  (T, E, time_string(elapsed)), file=sys.stdout)
            sys.stdout.flush()
        else:
            remain = (self.steps - step) * (elapsed / step)
            print(header)
            print('%12.5f  %12.2f  %7.2f%%  %7.2f%%  %s  %s' %
                  (T, E, 100.0 * acceptance, 100.0 * improvement,
                   time_string(elapsed), time_string(remain)))
            sys.stdout.flush()

    def copy_state(self, state):
        return state.copy()


# Variant with an alternative codification and environment definition
class Alt1NQueensBaseAnnealer(NQueensBaseAnnealer):
    def __init__(self, n_queens=7, load_state=None):
        self.n_queens = n_queens
        init = np.random.randint(low=0, high=n_queens, size=n_queens)
        simanneal.Annealer.__init__(self,
                                    initial_state=init, load_state=load_state)
        self.reset_metrics()

    def move(self):
        self.epochs += 1
        # Select a random queen
        q = random.randint(0, self.n_queens - 1)
        # Move it to a different position in the same column
        r = random.randint(0, self.n_queens - 2)
        if r >= self.state[q]:
            r += 1
        self.state[q] = r


# Variant with an alternative codification and environment definition
class Alt2NQueensBaseAnnealer(Alt1NQueensBaseAnnealer):

    def move(self):
        self.epochs += 1
        # Select a random queen
        q = random.randint(0, self.n_queens - 1)
        # Move it to a random position forward or backward
        r = random.randint(0, 1) - 1
        self.state[q] = (self.state[q] + r) % self.n_queens


if __name__ == '__main__':
    N = 27
    ann = NQueensBaseAnnealer(n_queens=N)
    schedule = ann.auto(minutes=1)
    print()
    print(schedule)
    # schedule['updates'] = schedule['steps']

    ann = NQueensBaseAnnealer(n_queens=N)
    ann.set_schedule(schedule)
    best_sol, best_cost = ann.anneal()
    print()
    for i in range(N):
        for j in range(N):
            if best_sol[j] == i:
                print('X', end=' ')
            else:
                print('.', end=' ')
        print()
    print(best_cost)
    print('Epocas:' + str(ann.epochs))
