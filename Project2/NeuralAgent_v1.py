import sys

import pylab as plb
import numpy as np
import mountaincar

class  NeuralAgent():
    """A not so good agent for the mountain-car task.
    """

    def __init__(self, n_neurons,tau,eta,gamma,lambda_eligibility):
    
        self.mountain_car = mountaincar.MountainCar()

        # Parameters
        self.n_neurons = n_neurons
        self.tau = tau
        self.lambda_eligibility = lambda_eligibility
        self.gamma = gamma
        self.eta = eta

        # Defines the neural lattice
        self.neurons_pos = np.linspace(-150,30,n_neurons)
        print(self.neurons_pos)
        self.sigma_pos = self.neurons_pos[1]-self.neurons_pos[0]
        print(self.sigma_pos)
        self.neurons_vel = np.linspace(-15,15,n_neurons)
        print(self.neurons_vel)
        self.sigma_vel = self.neurons_vel[1]-self.neurons_vel[0]
        print(self.sigma_vel)
        self.pos_grid,self.vel_grid = np.meshgrid(self.neurons_pos,self.neurons_vel)

        # initialize the Q-values etc.
        self._init_run()



    def run(self,N_trials=10,N_runs=1,max_steps=1000000):     # Change N_runs and Reset when doing 10 agents run!
        self.latencies = np.zeros(N_trials)
        
        for run in range(N_runs):
            self.mountain_car.reset()
            self._init_run()
            latencies = self._learn_run(N_trials=N_trials,max_steps=max_steps)
            self.latencies += latencies/N_runs
            #call reset() to reset Q-values and latencies, ie forget all he learnt 
            # self.reset()
        
        return self.latencies

    def reset(self):
        """
        Reset the weights (and the latency_list).
        
        Instant amnesia -  the agent forgets everything he has learned before    
        """
        self.mountain_car.reset()
        self.weights = np.zeros((3,self.n_neurons,self.n_neurons))
        self.latency_list = []

    def _init_run(self):
        """
        Initialize the Q-values, eligibility trace, position etc.
        """
        # initialize the Q-values and the eligibility trace

        #Activations & Weights
        self.activations = np.zeros((self.n_neurons,self.n_neurons))
        self.weights = np.zeros((3,self.n_neurons,self.n_neurons)) # 3 because 3 outputs neurons_pos

        #Elegibility
        self.e = np.zeros((3,self.n_neurons,self.n_neurons))
        
        # list that contains the times it took the agent to reach the target for all trials
        # serves to track the progress of learning
        self.latency_list = []

        # initialize the state and action variables
        self.pos = None
        self.vel = None
        self.action = None


    def _learn_run(self,N_trials=10,max_steps=1000000):
        """
        Run a learning period consisting of N_trials trials. 
        
        Options:
        N_trials :     Number of trials

        Note: The Q-values are not reset. Therefore, running this routine
        several times will continue the learning process. If you want to run
        a completely new simulation, call reset() before running it.
        
        """
        for trial in range(N_trials):
            # run a trial and store the time it takes to the target
            latency = self._run_trial(max_steps)
            self.latency_list.append(latency)

        return np.array(self.latency_list)


    def _run_trial(self,max_steps):
        """
        Run a single trial until the agent reaches the reward position.
        Return the time it takes to get there.

        Options:
        visual: If 'visualize' is 'True', show the time course of the trial graphically
        """
        # reset the mountain_car
        self.mountain_car.reset()

        # choose the initial position
        self.pos = np.random.uniform(-130, -50)
        self.pos_old = None
        self.vel = np.random.uniform(-5,5)
        
        print("Starting trial at position ({0},{1})".format(self.pos,self.vel))

        # initialize the latency (time to reach the target) for this trial
        latency = 0

        # run the trial
        self._choose_action()
        while (not self._arrived()) and (latency<max_steps):
            self._update_state()
            self._choose_action()    
            self._update_weights()
        
            latency = latency + 1

        print(self.weights.sum(axis=1).sum(axis=1))
        print(latency)
        return latency


    def _arrived(self):
        #if self.pos_old != None:
        return (self.pos>0) #Exit condition.


    def Softmax(self,Q):
        expo = np.exp(Q/self.tau)
        probs = expo/expo.sum()
        return probs
        

    def _choose_action(self):    
        """
        Softmax on Q-values and random chosing
        """
        self.action_old = self.action
        self.activations_old = self.activations
        self.activations = np.exp(-(np.square(self.pos - self.pos_grid)/self.sigma_pos)-(np.square(self.vel - self.vel_grid)/self.sigma_vel))

        self.Q = np.multiply(self.weights, self.activations).sum(axis=1).sum(axis=1)
        probabilities = self.Softmax(self.Q)

        self.action = np.random.choice([0,1,2],size=1,p=probabilities)
        
    def _update_state(self):
        '''
        Performs the current action
        '''
        self.pos_old = self.pos
        self.vel_old = self.vel
        self.mountain_car.apply_force(self.action)
        self.mountain_car.simulate_timesteps(100, 0.01)
        self.pos=self.mountain_car.x
        self.vel=self.mountain_car.x_d


    def _update_weights(self):
        """
        Update the current estimate of the Q-values / weights, according to SARSA.
        """
        
        # update the weights
        if self.action_old != None:
            
            q_old = np.multiply(self.weights, self.activations_old).sum(axis=1).sum(axis=1)
            q_new = np.multiply(self.weights, self.activations).sum(axis=1).sum(axis=1)
            delta_t = self.mountain_car.R - (q_old[self.action_old] - self.gamma * q_new[self.action])
            self.e_old = self.e
            self.e[self.action_old] = self.gamma * self.lambda_eligibility * self.e_old[self.action_old] + self.activations_old
            delta_weights = self.eta * delta_t * self.e[self.action_old]
            self.weights[self.action_old] += delta_weights



    def visualize_trial(self, n_steps = 200):
        """Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """
        
        # prepare for the visualization
        plb.ion()
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()
            
        # make sure the mountain-car is reset
        self.mountain_car.reset()

        for n in range(n_steps):
            # For time step showing
            #print('\rt =', self.mountain_car.t)
            #sys.stdout.flush()
            
            # choose action
            self._choose_action()
            self.mountain_car.apply_force(self.action)
            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)

            # update the visualization
            mv.update_figure()
            plb.draw()            
            
            # check for rewards
            if self.mountain_car.R > 0.0:
                print ("\rreward obtained at t = ", self.mountain_car.t)
                break

if __name__ == "__main__":
    
    n_neurons = 20 # Actually a n_neuros x n_neurons grid
    tau = 1 #Exploration temperature parameter
    eta = 0.1 #Learning rate
    gamma = 0.95 #Reward factor
    lambda_eligibility = 0.95 #elegibility decay
    a = NeuralAgent(n_neurons,tau,eta,gamma,lambda_eligibility)

    N_trials = 20
    N_runs=1
    max_steps = 10000 #Save stop if not working well
    latencies = a.run(N_trials,N_runs,max_steps)
    print(latencies)
