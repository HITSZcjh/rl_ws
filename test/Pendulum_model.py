from casadi import *
from acados_template import AcadosSim, AcadosSimSolver, AcadosModel
import math
import time

class PendulumModel(object):
    def __init__(self, dt=0.01) -> None:

        m = 0.2
        M = 1.0
        g = 9.81
        l = 1.0

        model = AcadosModel()
        theta = SX.sym("theta")
        omega = SX.sym("omega")
        x = SX.sym("x")
        v = SX.sym("v")

        f = SX.sym("f")

        model.x = vertcat(x,v,theta,omega)

        v_dot = (f+m*sin(theta)*(g*cos(theta)-omega**2*l))/(M+m*sin(theta)**2)
        omega_dot = (g*sin(theta)+v_dot*cos(theta))/l
        model.f_expl_expr = vertcat(
            v,
            v_dot,
            omega,
            omega_dot
        )

        x_dot = SX.sym("x_dot", 5)
        model.xdot = x_dot
        model.u = f
        model.p = []
        model.name = "PendulumModel"

        sim = AcadosSim()
        sim.model = model
        sim.solver_options.T = dt
        sim.solver_options.integrator_type = "ERK"
        sim.solver_options.num_stages = 3
        sim.solver_options.num_steps = 3
        sim.solver_options.newton_iter = 3  # for implicit integrator
        sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"

        self.acados_integrator = AcadosSimSolver(sim)

        self.action_dim = 1
        self.action_range = [-10.0, 10.0]
        self.state_dim = 4 
        self.state = np.zeros(self.state_dim)
        self.reward = 0
        self.max_ep_len = 1000
        self.itr_num = 0
        self.done = 0
        
        self.acados_integrator.set("x", np.zeros(self.state_dim))

    def is_state_available(self):
        if(math.fabs(self.state[0])>5.0 or math.fabs(self.state[2])>math.pi/4.0):
            return False
        else:
            return True
        
    def reset(self, state:np.ndarray = None):
        if isinstance(state, np.ndarray):
            self.state = state
        else:
            self.state = np.zeros(self.state_dim)
            self.state[2] = math.pi/4*(np.random.random()-0.5)
        
        if(not self.is_state_available()):
            print("Initial state error!")
            exit()

        self.done = 0
        self.itr_num = 0
        self.acados_integrator.set("x", self.state) 

        return self.state
    
    def step(self, action:np.ndarray):
        if self.done == 1:
            print("env is done!")
            exit()
        if action > self.action_range[1]:
            action = self.action_range[1]
        elif action < self.action_range[0]:
            action = self.action_range[0]
            
        self.acados_integrator.set("x", self.state)
        self.acados_integrator.set("u", action)
        status = self.acados_integrator.solve()

        if (status != 0):
            print("Integate error!")
            exit()

        self.state = self.acados_integrator.get("x")
        self.itr_num += 1

        if(self.is_state_available()):
            self.reward = 1
        else:
            self.reward = -100

        if(not self.is_state_available() or self.itr_num > self.max_ep_len):
            self.done = 1
        
        return self.state, self.reward, self.done
    
if __name__ == "__main__":

    env = PendulumModel()
    temp = env.reset(np.array([0,0,math.pi/16,0]))
    start_time = time.time()
    for i in range(100):
        if env.state[2]>0:
            temp = env.step(-10)
        else:
            temp = env.step(10)       
    end_time = time.time()
    run_time = end_time - start_time

    print("程序运行时间（秒）:", run_time)