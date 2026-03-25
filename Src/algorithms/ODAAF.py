import numpy as np
import random as rd

class ODAAF():

    def __init__(self, arms = None):
        self.ground_arms = arms
        self.arms_pool = self.ground_arms.copy()
        self.name = "ODAAF"
        self.arms_payoff_vectors = {"cumulated_rewards" : np.zeros(len(self.ground_arms)),
                                    "tries" : np.zeros(len(self.ground_arms))
                                    }
        self.arm_chosen = None
        self.threshold = 4
        self.delta = 0.5
        #Aggregated anonymous feedback
        self.X = []
        #Arms played history 
        self.Tj = []
        for arm in self.ground_arms["arm_id"]:
            #For each arm we have an history of each iteration when the arm has been played
            self.Tj.append([])
        self.C1, self.C2 = 1, 1
        #Majoring distribution
        self.d = 1


    def run(self, observed_value, user_context=None):
        self.init_choice(observed_value)
        self.arm_chosen = self.choose_action()
        
        return self.arm_chosen
    

    def init_choice(self, observation):
        self.arm_chosen = -1
        self.arms_pool = self.ground_arms[self.ground_arms["arm_id"].isin(observation["arm_id"])]
        self.arms_pool.reset_index(inplace=True)

    
    def stepOne(self, m, t, horizon, results):
        """
        Play arms
        """
        term1 = (self.C1*np.log(horizon*self.delta**2))/(self.delta**2)
        term2 = (self.C2*m *self.d)/self.delta
        nm = int(max(1, term1 + term2))
        for arm_id in self.arms_pool["arm_id"]:
            while len(self.Tj[arm_id]) < nm :
                observed_value = rd.choice(results[results["arm_id"] == arm_id]).copy()
                observed_delay = self.get_delay(observed_value["arm_id"].iloc[0])
                self.Tj[arm_id].append(t)
                t+=1
                

    def stepTwo(self):
        """
        eliminate suboptimal arms
        """
        pass
    def stepThree(self):
        """
        Decrease tolerance
        """
        pass
    def stepFour(self):
        """
        Bridge period
        """
        pass
        

    
    def choose_action(self):
        arm_chosen_index = -1

        if np.min(self.arms_payoff_vectors["tries"]) == 0 :
            i=0
            for arm in self.arms_pool['arm_id']:
                arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == arm]

                if (self.arms_payoff_vectors["tries"][arm_pos] < 1) :
                    arm_chosen_index = i
                    break

                i += 1

        if arm_chosen_index == -1 :

            arm_pool_size = len(self.arms_pool['arm_id'])
            expected_payoff = np.zeros(arm_pool_size) - 1
            i=0
            for arm in self.arms_pool['arm_id']:
                arm_pos = np.where(self.ground_arms["arm_id"] == arm)[0][0]
                expected_payoff[i] = self.arms_payoff_vectors["cumulated_rewards"][arm_pos] / self.arms_payoff_vectors["tries"][arm_pos]
                i += 1 

            arm_chosen_index = np.argmax(expected_payoff + np.sqrt((2*np.log(np.sum(self.arms_payoff_vectors["tries"]))) / self.arms_payoff_vectors["tries"]))
        arm_chosen = self.arms_pool["arm_id"][arm_chosen_index]
            
        return arm_chosen
    

    def evaluate(self, observation):
        reward = 0
        feedback = observation["feedback"][observation["arm_id"] == self.arm_chosen].iloc[0]
        if feedback >= self.threshold:
            reward = 1

        return reward


    def update(self, observation):
        observed_reward = self.evaluate(observation)
        self.arms_payoff_vectors["cumulated_rewards"][self.arm_chosen] += observed_reward
        self.arms_payoff_vectors["tries"][self.arm_chosen] += 1
