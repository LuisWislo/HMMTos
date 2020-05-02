import numpy as np
import pandas as pd
import detectos

def viterbi(pi, a, b, obs):
    
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]
    
    # init blank path
    path = path = np.zeros(T,dtype=int)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))
    
    # init delta and phi 
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    print('\nStart Walk Forward\n')    
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]] 
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))
    
    # find optimal path
    print('-'*50)
    print('Start Backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    for t in range(T-2, -1, -1):
        path[t] = phi[path[t+1], [t+1]]
        print('path[{}] = {}'.format(t, path[t]))
        
    return path, delta, phi

obs_map = {'Low':0, 'Mid':1, 'High':2}

# Secuencia observada
# Esto se sacaria de un .wav, yo cree esta secuencia de prueba
obs = np.array(detectos.getObservables('cough3.wav'))

inv_obs_map = dict((v,k) for k, v in obs_map.items())
obs_seq = [inv_obs_map[v] for v in list(obs)]

print("Simulated Observations:\n",pd.DataFrame(np.column_stack([obs, obs_seq]),columns=['Obs_code', 'Obs_seq']) )

states = ['Low', 'Mid', 'High']
hidden_states = ['A', 'B', 'C', 'D', 'E']

# Probabilidades iniciales
pi = [0, 0, 0, 0, 1]

state_space = pd.Series(pi, index=hidden_states, name='states')
a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)

# Matriz de transiciones con las probabilidades
a_df.loc[hidden_states[0]] = [0.2, 0.8, 0, 0, 0]
a_df.loc[hidden_states[1]] = [0, 0.5, 0.5, 0, 0]
a_df.loc[hidden_states[2]] = [0, 0, 0.2, 0.8, 0]
a_df.loc[hidden_states[3]] = [0.3, 0, 0, 0.3, 0.4]
a_df.loc[hidden_states[4]] = [0.2, 0, 0, 0, 0.8]

print("\n HMM matrix:\n", a_df)
a = a_df.values

observable_states = states
b_df = pd.DataFrame(columns=observable_states, index=hidden_states)

# Matriz de las probabilidades de observacion con relacion a un estado
b_df.loc[hidden_states[0]] = [0,0.1,0.9]
b_df.loc[hidden_states[1]] = [0,0.8,0.2]
b_df.loc[hidden_states[2]] = [0,0.6,0.4]
b_df.loc[hidden_states[3]] = [1,0,0]
b_df.loc[hidden_states[4]] = [1,0,0]

print("\n Observable layer  matrix:\n",b_df)
b = b_df.values

path, delta, phi = viterbi(pi, a, b, obs)
state_map = {0:'Snow', 1:'Rain', 2:'Sunshine'}
state_path = [state_map[v] for v in path]
pd.DataFrame().assign(Observation=obs_seq).assign(Best_Path=state_path)
