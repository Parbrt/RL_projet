'''
Created on 23 mars 2024

@author: aletard
'''

#----------------------------------------------------------------#
#                                                                #
#                    External imports                            #
#                                                                #
#----------------------------------------------------------------#


import time
import numpy as np
import random as rd


#----------------------------------------------------------------#
#                                                                #
#                    Packages imports                            #
#                                                                #
#----------------------------------------------------------------#


from Src.utils.repository_manager import RepositoryManager as RM
from Src.data_management.data_loader import DataLoader as DL
from Src.Reporting.report_generator import ReportGenerator
from Src.Reporting.results_storer import ResultStorer

from Src.algorithms.EGreedy import EGreedy
from Src.algorithms.Random import Random
from Src.algorithms.UCB import UCB
from Src.algorithms.ODAAF import ODAAF
from Src.algorithms.Thompson import Thompson
from Src.algorithms.linUCB import linUCB



#----------------------------------------------------------------#
#                                                                #
#                    Global variables                            #
#                                                                #
#----------------------------------------------------------------#


# ... No global variables defined ...


#----------------------------------------------------------------#
#                                                                #
#                    Abstract Classes                            #
#                                                                #
#----------------------------------------------------------------#




#----------------------------------------------------------------#
#                                                                #
#                    Functions & Classes                         #
#                                                                #
#----------------------------------------------------------------#


class SimulatorODAAF():
    """
    Implement the Cumulated Anonymous Delay for the ODAAF algorithme.
    """
    
    def __init__(self):
        
        print("Initializing Simulator")
        
        
        # datas is a dictionnary of dataframe
        self.dataset_name = "03-RSASM"
        self.datas = self.data_extraction()
        # Delayed distribution for each arm
        self.datas["arms"]["delay_mean"] = np.zeros(len(self.datas["arms"]))
        self.datas["arms"]["delay_var"] = np.zeros(len(self.datas["arms"]))
        
        #We assign a distribution of delay for each arm in the dataset
        for index,arm in self.datas["arms"].iterrows() :
            self.datas["arms"].loc[index, "delay_mean"] = rd.randint(3, 10)
            self.datas["arms"].loc[index, "delay_var"] = rd.randint(0, 3)

        self.datas["results"]["delay"] = np.zeros(len(self.datas["results"]))
        print("Computing delays for each observation...")
        
        for index, result in self.datas["results"].iterrows() :
            self.datas["results"].loc[index,"delay"] = self.get_delay(self.datas["results"].loc[index,"arm_id"])
            
        self.horizon = 30000
        self.algorithm = ODAAF(self.datas["arms"], self.horizon)
        
        self.results = ResultStorer(self.horizon)
        self.reporter = ReportGenerator(RM.create_repository_with_timestamp("../Output"), \
                                        (self.dataset_name, self.horizon, self.algorithm.name))
        
        # 1st parameter for time in seconds, 2nd for number of iterations
        self.life_sign_delay = (300, 5000)
        
        
        #-----------------------
    def get_delay(self, arm_id):
        """
        Get delay for given arm
        """
        delay = np.random.normal(self.datas["arms"]["delay_mean"][arm_id], self.datas["arms"]["delay_var"][arm_id])
        return int(max(0, delay)) 

    def run_simulation(self):

        print("starting simulation")
        print(self.datas["results"].head())        
        
        self.results.start_time = time.time()
        m = 0
        t = 1
        
        while t < self.horizon:
            m+=1
            last_t = t
            print(f"Entering phase :{m}\n Iteration :{t}")
            t , X= self.algorithm.run(m, t, self.datas["results"])
           
           
            #We log each reward generated during one phase inside the algorithm
            for i in range(last_t,t):
                self.results.update_measures_v2(i, X[i])
            self.sign_life(t-1)
            
            
        self.results.end_time = time.time()
        self.end_sign()

        #-----------------------

    def data_extraction(self):

        rss_path = RM.get_absolute_from_relative_path(f"../Resources/bandit_datasets/{self.dataset_name}")
        files_to_load = RM.get_files_in_directory(rss_path)
        files_path = []
        for file in files_to_load :
            files_path.append(f"{rss_path}/{file}")
            
        return DL.load_multiple_files(files_path)

        #-----------------------

    def context_formatter(self, context):
        
        try :
            context = context.drop(["context_id"], axis=1)        
        except:
            print("Error on context formatting")
        
        nb_dimensions = context.shape[1]
        context = np.array(context)
        user_context = context.reshape(nb_dimensions, )
        
        
        return user_context

        #-----------------------

    def sign_life(self, iteration):

        sign_life_message = f"\nSimulator has been running for {round(time.time() - self.results.start_time, 3)} seconds. \n" + \
                               f"Currently going for iteration {iteration}, latest accuracy value : {round(100*np.sum(self.results.algorithm_performance["cumulated_reward"])/iteration+1,2)}," + \
                               f" cumulated regrets: {round(self.results.algorithm_performance['cumulated_regrets'][iteration],3)}.\n\n" \
                                   

        self.reporter.log_generator(sign_life_message)
        
        #-----------------------
        
    def end_sign(self):

        end_message = f"\nSimulation correctly ended. \n The simulation has been running for {round(self.results.end_time - self.results.start_time, 3)} seconds. \n" + \
                        f"The simulation included {self.horizon} iterations,latest accuracy value : {round(100*np.sum(self.results.algorithm_performance["cumulated_reward"])/self.horizon,2)}," + \
                        f" cumulated regrets: {round(self.results.algorithm_performance['cumulated_regrets'][self.horizon-1],3)}.\n\n"

        self.reporter.log_generator(end_message)
        
        
    #----------------------------------------------------------------------------------------- 
    
    