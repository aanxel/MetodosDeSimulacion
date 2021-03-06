# Ruleta (37 ranuras) del 0 al 36
#   Del 1 al 10 y del 19 al 28 los número pares son rojos y los impares negros
#   Del 11 al 18 y del 29 al 36 los número pares son negros y los impares rojos
#   El 0 es verde. (House Edge)

# Pagos 1:1 -> 18/37 (Par/Imapar, Negro/Rojo, Alto/Bajo)
# Pagos 2:1 -> 12/37 (Serie de 12)
# Pagos 5:1 -> 6/37 (Serie de 6)
# Pagos 8:1 -> 4/37 (Serie de 4)
# Pagos 11:1 -> 3/37 (Serie de 3)
# Pagos 17:1 -> 2/37 (Serie de 2)
# Pagos 35:1 -> 1 / 37 (Un único número)

# Restricciones
#   Apueta una única ficha
#   Retirarse cuando no te quedan fichas o alcanzas 150
#   Máximo 50 partida
import numpy as np
import numpy.random as npr
import pickle
import math
import time
import signal
import sys
import matplotlib.pyplot as plt
import simanneal
from simanneal.anneal import time_string
from IPython.display import clear_output
import multiprocessing as mp
import datetime

def handler(*args):
    pass

def init_worker():
    signal.signal(signal.SIGINT, handler)

class SimCasino:
    jugadas = np.array([18/37, 12/37, 6/37, 4/37, 3/37, 2/37, 1/37])
    ganancias_jugadas = (36 / (jugadas * 37)).astype(int) - 1

    def __init__(self, n_dias=30, max_partidas=50, max_fichas=150,
                 fichas_inicial=30, probs_jugadas=[1/7]*7):
        self.n_dias = n_dias
        self.max_partidas = max_partidas
        self.max_fichas = max_fichas
        self.fichas_inicial = fichas_inicial
        self.probs_jugadas = probs_jugadas

        
    def compilar_probabilidades(self):
        "Compilar array de probabilidades para seleccion mas eficiente"
        p_jugadas = list(enumerate(self.probs_jugadas))
        p_jugadas.sort(key=lambda e: -1* e[1])  # Ordenar por prob mayor a menor
        # Probabilidades ordenadas de mayor a menor
        self.probs_jugadas_f = np.array([e[1] for e in p_jugadas])
        # Asociacion a jugada original dado el indice en el array ordenado
        self.jugada_f_a_jugada = np.array([e[0] for e in p_jugadas])
        
    # Selecciona una jugada según probs_jugadas y devuelve el beneficio
    def realizar_jugada(self):
        i_jugada = npr.choice(len(self.jugadas), p=self.probs_jugadas)
        if npr.random() < self.jugadas[i_jugada]:
            return self.ganancias_jugadas[i_jugada]
        else:
            return -1
        
    def realizar_jugada_fast(self):
        "Versión que emplea un vector de probabilidades ordenado"
        "Requiere una llamada previa a compilar_probabilidades()"
        # Elegir jugada de forma más eficiente
        # https://tinyurl.com/2p8p4hp3
        x = npr.random()
        cum = 0
        for i_jugada, p in enumerate(self.probs_jugadas_f):
            cum += p
            if x < cum:
                break
        i_jugada = self.jugada_f_a_jugada[i_jugada]
        if npr.random() < self.jugadas[i_jugada]:
            return self.ganancias_jugadas[i_jugada]
        else:
            return -1

    def simular_n_dias(self):
        # Almacena el total de fichas al final de cada día
        fichas_dia = np.zeros((self.max_fichas + self.ganancias_jugadas[-1]))
        # Almacena total de días en bancarrota
        n_bancarrotas = 0
        # Número total de partidas jugadas cada dia terminado en bancarrota
        acc_partidas_bancarrota = 0

        for _ in range(self.n_dias):
            # Repostar fichas
            fichas = self.fichas_inicial
            # Número de partidas jugadas hoy
            partidas_hoy = 0
            while (fichas > 0 and fichas < self.max_fichas and
                   partidas_hoy < self.max_partidas):
                # Jugar una partida
                fichas += self.realizar_jugada_fast()
                partidas_hoy += 1
            fichas_dia[fichas] += 1
            if fichas == 0:
                n_bancarrotas += 1
                acc_partidas_bancarrota += partidas_hoy

        if n_bancarrotas != 0:
            p_part_antes_bancarrota = acc_partidas_bancarrota / n_bancarrotas
        else:
            p_part_antes_bancarrota = 0
        return (n_bancarrotas / self.n_dias,
                fichas_dia,
                p_part_antes_bancarrota)

    class SimularAux(object):
        def __init__(self, simulador):
            self.simulador = simulador

        def __call__(self, _):
            return self.simulador.simular_n_dias()

    def simular(self, n_simulaciones=1000, pool=None):
        self.compilar_probabilidades()
        fichas_dia = np.zeros((self.max_fichas + self.ganancias_jugadas[-1]))
        p_bancarrotas = 0
        p_part_antes_bancarrota = 0
        n_sims_con_bancarrota = 0
        if pool == None:
            pool = mp.Pool(mp.cpu_count(), init_worker)
        results = pool.map(__class__.SimularAux(self), range(0, n_simulaciones))
        for m_p_banc, m_f_dia, m_p_part_prev_bancarrota in results:
            fichas_dia += m_f_dia
            p_bancarrotas += m_p_banc
            if m_p_part_prev_bancarrota != 0:
                p_part_antes_bancarrota += m_p_part_prev_bancarrota
                n_sims_con_bancarrota += 1

        if n_sims_con_bancarrota != 0:
            p_part_antes_bancarrota /= n_sims_con_bancarrota
        else:
            p_part_antes_bancarrota = -1
        return (p_bancarrotas / n_simulaciones,
                fichas_dia / n_simulaciones,
                p_part_antes_bancarrota)

    class SimularAuxFast(object):
        def __init__(self, simulador):
            self.simulador = simulador

        def __call__(self, _):
            return self.simulador.simular_n_dias_fast()

    def simular_n_dias_fast(self):
        "Versión que solo calcula promedio de fichas"
        # Almacena el total de fichas al final de cada día
        fichas_dia = 0
        for _ in range(self.n_dias):
            # Repostar fichas
            fichas_hoy = self.fichas_inicial
            # Número de partidas jugadas hoy
            partidas_hoy = 0
            while (fichas_hoy > 0 and fichas_hoy < self.max_fichas and
                   partidas_hoy < self.max_partidas):
                # Jugar una partida
                fichas_hoy += self.realizar_jugada_fast()
                partidas_hoy += 1
            fichas_dia += fichas_hoy
        return fichas_dia / self.n_dias
 

    def simular_fast(self, n_simulaciones=1000, pool=None):
        "Versión de simular que solo calcula el promedio de fichas"
        self.compilar_probabilidades()
        if pool is None:
            pool = mp.Pool(mp.cpu_count(), init_worker)
        results = pool.map(__class__.SimularAuxFast(self),
                           range(0, n_simulaciones))
        return sum(results) / len(results)

    def matriz_transiciones(self, probs):
        n_estados = self.max_fichas + self.ganancias_jugadas[-1]
        M = np.zeros((n_estados, n_estados))
        p_perder = sum((1 - self.jugadas[i])*probs[i]
                       for i in range(len(self.jugadas)))
        ps_ganar = [self.jugadas[i]*probs[i]
                    for i in range(len(self.jugadas))]
        for i in range(n_estados):
            # Estados de abandono
            if i == 0 or i >= self.max_fichas:
                M[i][i] = 1
            else:
                for j in range(len(probs)):
                    M[i][i + self.ganancias_jugadas[j]] = ps_ganar[j]
                M[i][i-1] = p_perder
        return M

    def simular_m_transiciones(self):
        M = self.matriz_transiciones(self.probs_jugadas)
        s_0 = np.zeros(M.shape[0])
        s_0[self.fichas_inicial] = 1
        M_s = np.dot(s_0, np.linalg.matrix_power(M, self.max_partidas))
        return M_s, sum(p * f for f,p in enumerate(M_s))


class CasinoAnnealer(simanneal.Annealer):
    def __init__(self, initial_state=[1/7]*7, n_simulaciones=10000,
                 T_config={'L': 1}, stop_config={'p_acc': 0.1, 'k': 5},
                 load_state=None, desplazamiento=0.2, threads=None,
                 save_state=True):
        self.save_state_on_exit = save_state
        self.T_config = T_config
        self.stop_config = stop_config
        self.n_simulaciones = n_simulaciones
        self.threads = threads
        self.pool = None
        self.casino = SimCasino()
        self.desplazamiento = desplazamiento  # Define tamaño del entorno
        super().__init__(initial_state=initial_state, load_state=load_state)
        self.T_configuration()
        self.reset_metrics()

    def reset_metrics(self):
        self.epochs = 0
        self.T_hist = []
        self.E_hist = []
        self.best_E_hist = []
        self.accept_hist = []
        self.improv_hist = []
        self.steps_hist = []
        self.stop_config['trials'] = 0
        self.stop_config['accepts'] = 0

    def energy(self):
        self.casino.probs_jugadas = self.state
        if self.n_simulaciones:
            return -1* self.casino.simular_fast(self.n_simulaciones, self.pool)
        return -1 * self.casino.simular_m_transiciones()[1]

    def move(self):
        self.epochs += 1
        a = npr.randint(0, len(self.state))
        # Modificar a un número aleatorio y normalizar
        self.state[a] = npr.uniform(0, 1)
        self.state /= np.sum(self.state)

    def T_function(self):
        if self.epochs % self.T_config['L'] == 0:
            T = self.Tmax * self.T_config['alfa']**self.epochs
        else:
            T = self.Tmax * self.T_config['alfa']**(self.epochs - self.epochs % self.T_config['L'])
        return T

    def T_configuration(self):
        Tfactor = -math.log(self.Tmax / self.Tmin)
        self.T_config['alfa'] = math.exp(Tfactor / self.steps)
        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')

    def stop_criterion(self):
        if self.epochs % (self.T_config['L'] * self.stop_config['k']) == 0:
            if self.stop_config['accepts'] / self.stop_config['trials'] < self.stop_config['p_acc']:
                self.user_exit = True
                print('Se ha cumplido el criterio de parada por convergencia')
            self.stop_config['trials'] = 0
            self.stop_config['accepts'] = 0

    def anneal(self, updates=None):
        if updates:
            self.updates = updates
        ###
        self.reset_metrics()
        self.T_configuration()
        ###
        step = self.epochs
        # Note initial state
        self.best_state = self.copy_state(self.state)
        self.best_energy = np.inf
        return self.resume_anneal(step)

    def resume_anneal(self, step):
        signal.signal(signal.SIGINT, self.set_user_exit)
        self.pool = mp.Pool(self.threads, init_worker)
        self.start = time.time()
        E = self.energy()
        prevEnergy = E 
        self.best_energy = min(E, self.best_energy)
        prevState = self.copy_state(self.state)
        T = -1
        if self.updates > 0:
            updateWavelength = self.steps / self.updates
            self.update(step, T, E, -1, -1)
        trials, accepts, improves = 0, 0, 0
        self.user_exit = False
        # Attempt moves to new states
        while step < self.steps and not self.user_exit:
            step += 1
            ### Actualización de temperatura
            T = self.T_function()
            ###
            dE = self.move()
            if dE is None:
                E = self.energy()
                dE = E - prevEnergy
            else:
                E += dE
            trials += 1
            self.stop_config['trials'] += 1
            if dE > 0.0 and math.exp(-dE / T) < npr.random():
                # Restore previous state
                self.state = self.copy_state(prevState)
                E = prevEnergy
            else:
                # Accept new state and compare to best state
                accepts += 1
                self.stop_config['accepts'] += 1
                if dE < 0.0:
                    improves += 1
                prevState = self.copy_state(self.state)
                prevEnergy = E
                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E
            if self.updates > 1:
                if (step//updateWavelength) > ((step - 1)//updateWavelength):
                    self.update(
                        step, T, E, accepts / trials, improves / trials)
                    trials, accepts, improves = 0, 0, 0
            ###
            self.T_hist.append(T)
            self.E_hist.append(E)
            self.best_E_hist.append(self.best_energy)
            self.stop_criterion()
            ###
        if self.save_state_on_exit:
            self.save_state()  # Guardar estado de la clase
        print()
        # Return best state and energy
        return self.best_state, self.best_energy

    def save_state(self, fname=None):
        self.pool = None
        if not fname:
            date = datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
            fname = date + ".state"
        with open(fname, "wb") as fh:
            pickle.dump(self, fh)

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
                   (self.best_E_hist, 'Mejor Energía'),
                   (self.accept_hist, 'Aceptación'),
                   # (self.improv_hist, 'Mejora')
                   ]
        for i in range(2):
            for j in range(2):
                atr = metrics[i * 2 + j]
                f_ax = fig.add_subplot(gs[i, j])
                f_ax.set_title(f'{atr[1]}(iteraciones)')
                if (atr[1] == 'Energía' or atr[1] == 'Temperatura' or
                    atr[1] == 'Mejor Energía'):
                    f_ax.plot(list(range(0, self.epochs)), atr[0])
                    continue
                f_ax.plot(self.steps_hist, atr[0])
        plt.show()
        pass

    def default_update(self, step, T, E, acceptance, improvement):
        clear_output(wait=True)
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
