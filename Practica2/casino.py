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


class SimCasino:
    def __init__(self, n_dias=30, max_partidas=50, max_fichas=150,
                 fichas_inicial=30, probs_jugadas=[1/7]*7):
        self.n_dias = n_dias
        self.max_partidas = max_partidas
        self.max_fichas = max_fichas
        self.fichas_inicial = fichas_inicial
        self.probs_jugadas = probs_jugadas
        self.jugadas = [18/37, 12/37, 6/37, 4/37, 3/37, 2/37, 1/37]

    # Devuelve el beneficio de realizar una jugada
    def ganancia_jugada(self, p_ganar):
        if npr.random() < p_ganar:
            valor_ganancia = int(36 / (p_ganar * 37)) - 1
            return valor_ganancia
        else:
            return -1

    # Selecciona una jugada según probs_jugadas y añade el beneficio
    def realizar_jugada(self):
        i_jugada = npr.choice(len(self.jugadas), 1, p=self.probs_jugadas)[0]
        jugada = self.jugadas[i_jugada]
        return self.ganancia_jugada(jugada)

    def simular_n_dias(self):
        # Almacena el total de fichas al final de cada día
        fichas_dia = np.zeros((self.max_fichas + 35))
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
                fichas += self.realizar_jugada()
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

    def simular(self, n_simulaciones=1000):
        fichas_dia = np.zeros((self.max_fichas + 35))
        p_bancarrotas = 0
        p_part_antes_bancarrota = 0
        n_sims_con_bancarrota = 0
        for _ in range(n_simulaciones):
            m_p_banc, m_f_dia, m_p_part_prev_bancarrota = self.simular_n_dias()
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
