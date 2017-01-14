import sys
import time
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
        self.effective_tau = tau
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
    
    def _init_run(self):
        """
        Initialize the Q-values, eligibility trace, position etc.
        """

        # Activations, Weights, Eligibility
        self.weights = np.zeros((3,self.n_neurons,self.n_neurons)) # 3 because 3 outputs
        self.weights_multiplex = np.zeros((3,self.n_neurons,self.n_neurons)) # to avoid moving target within the run
        self.e = np.zeros((3,self.n_neurons,self.n_neurons))
        self.activations = np.zeros((self.n_neurons,self.n_neurons))
        self.Q = np.multiply(self.weights, self.activations).sum(axis=1).sum(axis=1)

        
        # list that contains the times it took the agent to reach the target for all trials
        # serves to track the progress of learning
        self.latency_list = []

        # initialize the state and action variables
        self.pos = None
        self.vel = None
        self.energy = None
        self.action = None



    def run(self,N_trials=10,N_runs=1,max_steps=1000000):     # Change N_runs and Reset when doing 10 agents run!
        self.latencies = np.zeros(N_trials)
        
        for run in range(N_runs):
            self.mountain_car.reset()
            self._init_run()
            latencies = self._learn_run(N_trials,max_steps)
            self.latencies += latencies/N_runs
            #call reset() to reset Q-values and latencies, ie forget all he learnt
        
        return self.latencies


    def _learn_run(self,N_trials=10,max_steps=1000000):
        """
        Run a learning period consisting of N_trials trials. 

        Note: The Q-values are not reset. Therefore, running this routine
        several times will continue the learning process. If you want to run
        a completely new simulation, call reset() before running it.
        
        """
        for trial in range(N_trials):
            
            # run a trial and store the time it takes to the target
            self.effective_tau = self.tau * np.exp(-trial/5)
            latency = self._run_trial(max_steps)
            self.latency_list.append(latency)

            # network multiplexing, only learning when it finishes
            if self._arrived():
                self.weights = self.weights_multiplex
            else:
                self.weights_multiplex = self.weights
                    

            print(self.weights.sum(axis=1).sum(axis=1)) # To check evolution of the weights


        return np.array(self.latency_list)


    def _run_trial(self,max_steps):
        """
        Run a single trial until the agent reaches the reward position.
        Return the time it takes to get there.

        """
        # reset the mountain_car
        self.mountain_car.reset()

        # choose the initial position
        self.pos = self.mountain_car.x
        self.pos_old = None
        self.vel = self.mountain_car.x_d
        self.vel_old = None
        self.energy = self.mountain_car._energy(self.pos,self.vel)
        self.energy_old = None
        
        print("Starting trial at position ({0},{1})".format(self.pos,self.vel))

        # reset activations and Elegibility
        self.activations = np.zeros((self.n_neurons,self.n_neurons))
        self.e = np.zeros((3,self.n_neurons,self.n_neurons))

        # initialize the latency (time to reach the target) for this trial
        self.latency = 0

        # run the trial
        self._choose_action()

        while (not self._arrived()) and (self.latency<max_steps):

            self._update_state()

            self._choose_action()
            
            self._update_weights()
            
            self.latency = self.latency + 1

        print(self.latency)
        return self.latency


    def _arrived(self):
        return (self.pos>0) #Exit condition.


    def Softmax(self,x):
        """
        Robust (normalized) verstion of Softmax
        Compute softmax values for each sets of scores in x.
        """
        x = x/self.effective_tau
        e_x = np.exp(x - np.max(x))
        probs = e_x / e_x.sum()
        return probs
        

    def _choose_action(self):    
        """
        Softmax on Q-values and random chosing
        """
        probabilities = self.Softmax(self.Q)

        self.action_old = self.action
        self.action = np.random.choice([0,1,2],size=1,p=probabilities)
        
    def _update_state(self):
        '''
        Performs the current action
        '''
        self.q_old = self.Q
        self.pos_old = self.pos
        self.vel_old = self.vel
        self.energy_old = self.energy

        self.mountain_car.apply_force(self.action-1)
        self.mountain_car.simulate_timesteps(100, 0.01)

        self.pos=self.mountain_car.x
        self.vel=self.mountain_car.x_d
        self.energy = self.mountain_car._energy(self.pos,self.vel)

        self.activations_old = self.activations
        self.activations = np.exp(-(np.square(self.pos - self.pos_grid)/(self.sigma_pos ** 2))-(np.square(self.vel - self.vel_grid)/(self.sigma_vel ** 2)))
        self.Q = np.multiply(self.weights, self.activations).sum(axis=1).sum(axis=1)


    def _update_weights(self):
        """
        Update the current estimate of the Q-values / weights, according to SARSA.
        """

        # update the weights
        if self.action_old != None:
            
            #delta_energy = self.energy-self.energy_old #reward for improving energy
            delta_t =  self.mountain_car.R - self.q_old[self.action_old] + self.gamma * self.Q[self.action] # + delta_energy

            self.e_old = self.e
            self.e = self.gamma * self.lambda_eligibility * self.e_old 
            self.e[self.action_old] = self.e[self.action_old] + self.activations_old
            
            delta_weights = self.eta * delta_t * self.e
            self.weights_multiplex += delta_weights



    def visualize_trial(self, n_steps, tau):
        """Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """

        print('Simulating for:')
        print(self.weights.sum(axis=1).sum(axis=1))
        
        # Initialize
        self.pos = None
        self.vel = None
        self.energy = None
        self.action = None
        self.activations = np.zeros((self.n_neurons,self.n_neurons))
        self.Q = np.multiply(self.weights, self.activations).sum(axis=1).sum(axis=1)

        self.tau = tau  #Let's us choose a lower temperature for evaluation


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
            
            self._choose_action()

            self._update_state()

            # update the visualization
            mv.update_figure()
            plb.draw()              
            
            # check for rewards
            if self.mountain_car.R > 0.0:
                print ("\rreward obtained at t = ", self.mountain_car.t)
                break
