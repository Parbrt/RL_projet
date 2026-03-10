import numpy as np



class Thompson():

    def __init__(self, arms = None):
        self.ground_arms = arms
        self.arms_pool = self.ground_arms.copy()
        self.name = "Thompson"
        self.arms_result_vectors = {"success" : np.zeros(len(self.ground_arms)),
                                    "fail" : np.zeros(len(self.ground_arms))}
        self.arm_chosen = None
        self.threshold = 4


    def run(self, observed_value, user_context=None):
        self.init_choice(observed_value)
        self.arm_chosen = self.choose_action()
        
        return self.arm_chosen
    

    def init_choice(self, observation):
        self.arm_chosen = -1
        self.arms_pool = self.ground_arms[self.ground_arms["arm_id"].isin(observation["arm_id"])]
        self.arms_pool.reset_index(inplace=True)

    
    def choose_action(self):
        arm_chosen_index = -1

        if arm_chosen_index == -1 :

            arm_pool_size = len(self.arms_pool['arm_id'])
            expected_payoff = np.zeros(arm_pool_size) - 1
            i=0
            for arm in self.arms_pool['arm_id']:
                arm_pos = np.where(self.ground_arms["arm_id"] == arm)[0][0]
                expected_payoff[i] = np.random.beta(self.arms_result_vectors["success"][arm_pos] + 1, self.arms_result_vectors["fail"][arm_pos] + 1)
                i += 1 

            arm_chosen_index = np.argmax(expected_payoff)
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
        if observed_reward == 1:
            self.arms_result_vectors["success"][self.arm_chosen] += 1
        else : 
            self.arms_result_vectors["fail"][self.arm_chosen] += 1
