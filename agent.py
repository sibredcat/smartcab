from __future__ import division
import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import os
import csv

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    
    
    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor
        
        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
     
        self. counter=1
        self.out=[]
        self.f = open(os.path.join('logs', 'stats.csv'), 'w')
    

    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """
        
        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        
        if testing==True: 
            self.epsilon=0
            self.alpha=0
        else:
            #self.epsilon=self.epsilon-0.05
            self.epsilon=math.exp(-0.01*self.counter)
            
            if self.alpha>-1/math.ceil(100*math.log(0.02)):
                self.alpha-=-1/math.ceil(100*math.log(0.02))
            else:
                self.alpha=0
            #print(math.ceil(100*math.log(0.02)))
        
        if self.counter==2:
            self.out.append(self.counter-1)
            self.out.append(self.env.success*1)
            self.out.append(self.net_reward)
            
            fieldnames = ['number', 'result','net_reward']
            writer=csv.DictWriter(self.f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'number':self.out[0],'result':self.out[1],'net_reward':self.out[2]})
            self.out=[]
            
        elif self.counter>2:
            self.out.append(self.counter-1)
            self.out.append(self.env.success*1)
            self.out.append(self.net_reward)
            
            fieldnames = ['number', 'result','net_reward']
            writer=csv.DictWriter(self.f, fieldnames=fieldnames)
            writer.writerow({'number':self.out[0],'result':self.out[1],'net_reward':self.out[2]})
            self.out=[]
        self.counter+=1
        self.net_reward=0
        
        
        
        
    
        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        
        # NOTE : you are not allowed to engineer eatures outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed 
        # constraints in order for you to learn how to adjust epsilon and alpha, and thus 
        #learn about the balance between exploration and exploitation.
        # With the hand-engineered features, this learning process gets entirely negated.
        
        # Set 'state' as a tuple of relevant data for the agent        
        state = (waypoint, inputs['light'], inputs['left'], inputs['oncoming'])
        
        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        
       
        
        maxQ = max(self.Q[state].values())
        return maxQ 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0

        if self.learning==True and state not in self.Q:
            self.Q[state]={None:0,'left': 0, 'right':0, 'forward':0}

        
        
        
        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None
        
        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        # Otherwise, choose an action with the highest Q-value for the current state
        # Be sure that when choosing an action with highest Q-value that you randomly select 
        #between actions that "tie".
        
        if self.learning==False:
            action=self.valid_actions[random.randint(0,3)]
            #num=1
        else:
            if self.epsilon>random.random():
                action=self.valid_actions[random.randint(0,3)]
                #num=2
            else:
                ind_list=[]
                [ind_list.append(ind) for ind, val in enumerate(self.Q[state].values()) if val == self.get_maxQ(state)]
                rand_action_index=ind_list[random.randint(0,len(ind_list)-1)]
                action=self.Q[state].keys()[rand_action_index]  
                #print(self.Q[state])
                #num=3
        #print(action, num)
        #print(self.epsilon)
        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')

        if self.learning:
            self.Q[state][action]=(1-self.alpha)*self.Q[state][action]+self.alpha*reward
            #print(self.Q)
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn
        
        self.net_reward=self.net_reward+reward
        #print 'net reward =', self.net_reward
        #print('status:', self.env.success)
        
        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """
    
    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent,learning=True,alpha=1.0,epsilon=1.0)
    
    for key in agent.Q:
        print key,
        print ["%0.2f" % i for i in agent.Q[key]]
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent,enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay=0.01 , log_metrics=True, optimized=True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test=10, tolerance=0.02)
    
     ## print Q table
    
    
if __name__ == '__main__':
    run()
