import numpy as np
import random as rd

class ODAAF():

    def __init__(self, arms = None, horizon=0):
        self.ground_arms = arms
        self.arms_pool = self.ground_arms.copy()
        self.name = "ODAAF"
        
        self.horizon = horizon
        
        self.arm_chosen = None
        self.threshold = 4
        self.delta = [0,0.5]
        self.min_delta = 1*10**(-10)
        #Aggregated anonymous feedback
        self.X = np.zeros(self.horizon)
        #Arms played history 
        self.Tj = []
        for arm in self.ground_arms["arm_id"]:
            #For each arm we have an history of each iteration when the arm has been played
            self.Tj.append([])
        self.C1, self.C2 = 1, 1
        #Majoring distribution
        self.d = np.mean(self.ground_arms["delay_mean"])


    def run(self,m, t, results):
        """
        Inputs:
            m: int phase index
            t: int iteration index
            results: dataframe Results data with delays and feedbacks
        """      
        
        nb_arms_left = len(self.arms_pool["arm_id"])
        print(f"{nb_arms_left} arms left")
        if nb_arms_left == 1:
            t = self.dumbStep(m, t, results)
        else :
            t = self.stepOne(m, t, results)
            
            if t >= self.horizon:
                print(f"Final iteration: {t}")
                
                return t, self.X
            
            self.stepTwo(m)
            
            self.stepThree()
            
            t = self.stepFour(m, t, results)
        
            
        return t, self.X
        


    def get_nm(self, m):
        """
        Compute the n value for phase m
        """
        
        term1 = (self.C1*np.log(self.horizon*self.delta[m]**2))/(self.delta[m]**2)
        term2 = (self.C2*m *self.d)/self.delta[m]
        
        return int(max(1, term1 + term2))
        
        
    
    def dumbStep(self, m, t, results):
        """
        Play arms carelessly
        """
        
        for arm_id in self.arms_pool["arm_id"]:
            
            self.arm_chosen =self.arms_pool["arm_id"][arm_id]
            
            print(f"running arm {arm_id} for {self.horizon-t} steps")
            
            while t < self.horizon:
                
                observed_context = rd.choice(results["context_id"])

                observed_value = results[(results["context_id"]== observed_context) & (results["arm_id"]== arm_id)]
                observed_delay = observed_value["delay"].iloc[0]
                
                #self.evaluate equivalent 
                observed_reward = 1 if observed_value["feedback"].iloc[0] >= self.threshold else 0
    
                index = int(observed_delay) + t
                
                if index < self.horizon :
                    self.X[index]+= observed_reward            
                
                self.Tj[arm_id].append(t)
                t+=1
        return t    
    
    def stepOne(self, m, t, results):
        """
        Play arms
        """
        nm = self.get_nm(m)
        
        for arm_id in self.arms_pool["arm_id"]:
            
            self.arm_chosen =self.arms_pool["arm_id"][arm_id]
            
            while len(self.Tj[arm_id]) < nm and t < self.horizon:
                
                observed_context = rd.choice(results["context_id"])

                observed_value = results[(results["context_id"]== observed_context) & (results["arm_id"]== arm_id)]
                observed_delay = observed_value["delay"].iloc[0]
                
                #self.evaluate equivalent 
                observed_reward = 1 if observed_value["feedback"].iloc[0] >= self.threshold else 0
    
                index = int(observed_delay) + t
                
                if index < self.horizon :
                    self.X[index]+= observed_reward            
                
                self.Tj[arm_id].append(t)
                t+=1
        return t
                

    def stepTwo(self, m):
        """
        eliminate suboptimal arms
        """
        meanX = np.zeros(len(self.ground_arms))
        
        for arm_id in self.arms_pool["arm_id"]:
            for t in self.Tj[arm_id]:
                
                meanX[arm_id] += self.X[t]
            meanX[arm_id] = meanX[arm_id]/len(self.Tj[arm_id])
                
        
        for arm_id in self.arms_pool["arm_id"]:
            if meanX[arm_id] + self.delta[m] < max(meanX):
                self.arms_pool.drop(self.arms_pool[self.arms_pool["arm_id"]== arm_id].index, inplace = True)
                print(f"Le bras {arm_id} a été supprimé !")
            
        
            
    def stepThree(self):
        """
        Decrease tolerance
        """
        self.delta.append(self.delta[-1]/2)
        #Avoids vanishing tolerence bound (div by 0)
        self.delta[-1] = self.min_delta if self.delta[-1] < self.min_delta else self.delta[-1] 
        
        print(f"Delta (tolerance) : {self.delta[-2]}")

    def stepFour(self, m, t, results): 
        """
        Bridge period
        """
        nm0 = self.get_nm(m)
        nm1 = self.get_nm(m - 1) if m > 1 else 0
        bridge_length = nm0 - nm1

        print(f"Bridge: {bridge_length} steps")

        bridge_arm = self.arms_pool["arm_id"].iloc[0]

        steps_done = 0
        while steps_done < bridge_length and t < self.horizon:

            observed_context = rd.choice(results["context_id"].unique().tolist())
            observed_value = results[(results["context_id"] == observed_context) & (results["arm_id"] == bridge_arm)]

            observed_delay = observed_value["delay"].iloc[0]
            observed_reward = 1 if observed_value["feedback"].iloc[0] >= self.threshold else 0

            index = int(observed_delay) + t
            if index < self.horizon:
                self.X[index] += observed_reward

            t += 1
            steps_done += 1

        return t if t < self.horizon else self.horizon
