'''
Created on 23 mars 2023

@author: aletard
'''

#----------------------------------------------------------------#
#                                                                #
#                    External imports                            #
#                                                                #
#----------------------------------------------------------------#


import matplotlib.pyplot as plt

#----------------------------------------------------------------#
#                                                                #
#                    Package imports                             #
#                                                                #
#----------------------------------------------------------------#

from Src.process.simulator_v2 import SimulatorODAAF
from Src.process.simulator_v3 import SimulatorAAD

#----------------------------------------------------------------#
#                                                                #
#                    Functions & Classes                         #
#                                                                #
#----------------------------------------------------------------#


HORIZON = 30000
OBSERVATIONS = 10

def main():
    simulator = SimulatorODAAF()
    simulator.run_simulation()
def test_difference() : 

    observations_ODAAF = {"accuracy":[],"regret":[]}
    observations_classic = {"accuracy":[],"regret":[]}

    #getting observations for the odaaf
    print("We run 10 times the alg ODAAF to gather metrics")
    for i in range(OBSERVATIONS):
        simulator = SimulatorODAAF()
        simulator.run_simulation()
        observations_ODAAF["accuracy"].append(round((simulator.horizon - simulator.results.algorithm_performance['cumulated_regrets'][-1])/simulator.horizon,2))
        observations_ODAAF["regret"].append(simulator.results.algorithm_performance['cumulated_regrets'])
    #getting observations for the classic alg
    print("We run 10 times the alg UCB (adaptated for Delayed Anonymous Agregated Rewards) to gather metrics")
    for i in range(OBSERVATIONS):
        simulator = SimulatorAAD()
        simulator.run_simulation()
        observations_classic["accuracy"].append(round((simulator.horizon - simulator.results.algorithm_performance['cumulated_regrets'][-1])/simulator.horizon,2))
        observations_classic["regret"].append(simulator.results.algorithm_performance['cumulated_regrets'])

    observations_ODAAF["mean_acc"] = sum(observations_ODAAF["accuracy"])/len(observations_ODAAF["accuracy"])
    observations_classic["mean_acc"] = sum(observations_classic["accuracy"])/len(observations_classic["accuracy"])
    print(f"mean accuracy for \nODAAF : {observations_ODAAF['mean_acc']}\nUCB : {observations_classic['mean_acc']}")
    observations_ODAAF["mean_reg"] = []
    observations_classic["mean_reg"] = []
    index = []
    for i in range(0,HORIZON,1000):
        index.append(i)
        observations_classic["mean_reg"].append(0)
        observations_ODAAF["mean_reg"].append(0)
        for j in range (OBSERVATIONS):
            observations_classic["mean_reg"][-1] += observations_classic["regret"][j][i]
            observations_ODAAF["mean_reg"][-1] += observations_ODAAF["regret"][j][i]
        observations_classic["mean_reg"][-1] = observations_classic["mean_reg"][-1] / OBSERVATIONS
        observations_ODAAF["mean_reg"][-1] = observations_ODAAF["mean_reg"][-1] / OBSERVATIONS
        
    print(f"mean final regret for \nODAAF : {observations_ODAAF['mean_reg'][-1]}\nUCB : {observations_classic['mean_reg'][-1]}")
    plt.plot(index, observations_classic["mean_reg"],label='UCB' )
    plt.legend()
    plt.plot(index, observations_ODAAF["mean_reg"],label='ODAAF' )
    plt.title('Mean Regret Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Regret')
    plt.show()
    
    
#--------------------------------------------------

if __name__ == "__main__":
    test_difference()
    