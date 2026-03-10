import numpy as np

class linUCB():

    def __init__(self, arms=None, alpha=1.0):
        self.ground_arms = arms
        self.arms_pool = self.ground_arms.copy()
        self.name = "linUCB"
        self.arm_chosen = None
        self.threshold = 4
        self.x_t_a = None     
        self.alpha = alpha
        self.A = {}            
        self.b = {}           

    def run(self, observed_value, user_context=None):
        self.x_t_a = user_context
        self.init_choice(observed_value)
        self.arm_chosen = self.choose_action()
        return self.arm_chosen

    def init_choice(self, observation):
        self.arm_chosen = -1
        self.arms_pool = self.ground_arms[self.ground_arms["arm_id"].isin(observation["arm_id"])].reset_index(drop=True)

        d = len(self.x_t_a)
        for arm in self.arms_pool["arm_id"]:
            if arm not in self.A:              
                self.A[arm] = np.identity(d)    
                self.b[arm] = np.zeros(d)       

    def choose_action(self):
        x = self.x_t_a.reshape(-1, 1)
        p = {}


        for arm in self.arms_pool["arm_id"]:
            A_a_inv = np.linalg.inv(self.A[arm])
            theta_a = A_a_inv @ self.b[arm]                        

            p[arm] = (theta_a @ self.x_t_a) + self.alpha * np.sqrt((x.T @ A_a_inv @ x).item())                                                       

        a_t = max(p, key=p.get)                               
        return a_t

    def evaluate(self, observation):
        feedback = observation["feedback"][
            observation["arm_id"] == self.arm_chosen
        ].iloc[0]
        r_t = 1 if feedback >= self.threshold else 0
        return r_t

    def update(self, observation):
        r_t = self.evaluate(observation)
        x = self.x_t_a.reshape(-1, 1)
        a_t = self.arm_chosen

        self.A[a_t] = self.A[a_t] + x @ x.T        
        self.b[a_t] = self.b[a_t] + r_t * self.x_t_a  