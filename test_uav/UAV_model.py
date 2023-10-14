from casadi import *
from acados_template import AcadosSim, AcadosSimSolver, AcadosModel
import math
import time
from scipy.spatial.transform import Rotation
A = 25647.0437921894
B = 0.47869017319
C = 0.000331169824804821
D = -277.900362668307
class UAVModel(object):
    def __init__(self, dt=0.01, max_ep_len=1000) -> None:
        # 系统状态
        p = SX.sym("p", 3, 1)
        v = SX.sym("v", 3, 1)
        w = SX.sym("w", 3, 1)
        q = SX.sym("q", 4, 1)
        f = SX.sym("f", 4, 1)
        # 系统状态集合
        state = vertcat(p, v, w, q, f)

        # 控制输入
        action = SX.sym("action", 4, 1)

        # 参数
        self.action_dim = action.size()[0]
        self.state_dim = state.size()[0]
        rotor_time_constant_up = 0.0125
        rotor_time_constant_down = 0.025
        Kf = 8.54858e-06  # rotot_motor_constant
        km = 0.016  # rotor_moment_constant
        body_length = 0.17
        mass = 0.716
        g = 9.81
        inertia = np.array([[0.007, 0, 0], [0, 0.007, 0], [0, 0, 0.012]])
        R = vertcat(
            horzcat(
                1 - 2 * (q[2] ** 2 + q[3] ** 2),
                2 * (q[1] * q[2] - q[0] * q[3]),
                2 * (q[1] * q[3] + q[0] * q[2]),
            ),
            horzcat(
                2 * (q[1] * q[2] + q[0] * q[3]),
                1 - 2 * (q[1] ** 2 + q[3] ** 2),
                2 * (q[2] * q[3] - q[0] * q[1]),
            ),
            horzcat(
                2 * (q[1] * q[3] - q[0] * q[2]),
                2 * (q[2] * q[3] + q[0] * q[1]),
                1 - 2 * (q[1] ** 2 + q[2] ** 2),
            ),
        )
        AllocationMatrix = np.array([[0, body_length, 0, -body_length],
                                     [-body_length, 0, body_length, 0],
                                     [km, -km, km, -km],
                                     [1, 1, 1, 1]])
        temp = AllocationMatrix@f
        F = vertcat(np.zeros([2, 1]), temp[3])
        Tao = temp[0:3]
        G = np.array([[0], [0], [-g]])
        q_dot = 1/2*vertcat(-q[1]*w[0]-q[2]*w[1]-q[3]*w[2],
                               q[0]*w[0]+q[2]*w[2]-q[3]*w[1],
                               q[0]*w[1]-q[1]*w[2]+q[3]*w[0],
                               q[0]*w[2]+q[1]*w[1]-q[2]*w[0])
        f_expl = vertcat(
            v,
            G+1/mass*R@F,
            np.linalg.inv(inertia)@(Tao-cross(w, inertia@w)),
            q_dot,
            1/rotor_time_constant_down*(action-f)
        )


        model = AcadosModel()
        x_dot = SX.sym('x_dot', self.state_dim, 1)
        model.x = state
        model.f_expl_expr = f_expl
        model.xdot = x_dot
        model.u = action
        model.p = []
        model.name = "UAVModel"

        sim = AcadosSim()
        sim.model = model
        sim.solver_options.T = dt
        sim.solver_options.integrator_type = "ERK"
        sim.solver_options.num_stages = 3
        sim.solver_options.num_steps = 3
        sim.solver_options.newton_iter = 3  # for implicit integrator
        sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"
        self.acados_integrator = AcadosSimSolver(sim)

        self.action_range = [0.0, 6.0]
        self.state = np.array([0,0,3,
                               0,0,0,
                               0,0,0,
                               1,0,0,0,
                               0,0,0,0])
        self.reward = 0
        self.max_ep_len = max_ep_len
        self.itr_num = 0
        self.done = 0

        self.acados_integrator.set("x", self.state)

    def is_state_available(self):
        # if np.any(np.abs(self.state[0:2])>5) or self.state[2]<0  or self.state[2]>6 or np.any(np.abs(self.state[6:9])>5):
        #     return False
        # else:
        #     return True
        
        if np.any(np.abs(self.state[0:3])>5) or np.any(np.abs(self.state[6:9])>5):
            return False
        else:
            return True

    def reset(self, state: np.ndarray = None):
        if isinstance(state, np.ndarray):
            self.state = state
        else:
            # self.state = np.zeros(self.state_dim)
            # random = np.random.random(15)
            # self.state[0:2] = 5*(random[0:2]-0.5)
            # self.state[2] = 3*(random[2]+0.5)
            # self.state[3:6] = 1*(random[3:6] - 0.5)
            # self.state[6:9] = 1*(random[6:9]-0.5)
            # pitch = math.pi / 4 *(random[9]-0.5)
            # roll = math.pi / 4 *(random[10]-0.5)
            # r = Rotation.from_euler("xyz",[0,pitch,roll],degrees=False)
            # self.state[9] = r.as_quat()[3]
            # self.state[10:13] = r.as_quat()[0:3]
            # self.state[13:17] = self.action_range[1]*(random[11:15])

            self.state = np.zeros(self.state_dim)
            random = np.random.random(15)
            self.state[0:3] = 5*(random[0:3] - 0.5)
            self.state[3:6] = 1*(random[3:6] - 0.5)
            self.state[6:9] = 1*(random[6:9] - 0.5)
            pitch = math.pi / 4 *(random[9]-0.5)
            roll = math.pi / 4 *(random[10]-0.5)
            r = Rotation.from_euler("xyz",[0,pitch,roll],degrees=False)
            self.state[9] = r.as_quat()[3]
            self.state[10:13] = r.as_quat()[0:3]
            self.state[13:17] = self.action_range[1]*(random[11:15])

        if not self.is_state_available():
            print("Initial state error!")
            exit()

        self.done = 0
        self.itr_num = 0
        self.acados_integrator.set("x", self.state)

        return self.state

    def get_reward(self):
        if not self.is_state_available():
            return -5000
        else:
            # distance = np.sum((self.state[0:3]-np.array([0,0,3]))**2)
            # omega_sum = np.sum(self.state[6:9]**2)
            # return (-4*distance+100)+0*(-8*omega_sum+100)

            distance = np.sum(np.abs(self.state[0:3]))
            omega_sum = np.sum(self.state[6:9]**2)
            if distance < 0.5 and np.sum(np.abs(self.state[3:6]))<0.5:
                return 2000 + (-20*distance+100)+0.25*(-omega_sum+100)
            else:
                return (-20*distance+100)+0.25*(-omega_sum+100)


            omega_sum = np.sum(self.state[6:9]**2)
            distance = np.sum(np.abs(self.state[0:3]))
            temp = math.pow(distance/C,B)
            return (A-D)/(1+temp)+D + (-omega_sum+100)

    
    def integrate_only(self, state: np.ndarray, action: np.ndarray):
        action_copy = action.copy()
        action_copy[action_copy>1] = 1
        action_copy[action_copy<0] = 0
        action_copy = action_copy*self.action_range[1]

        self.acados_integrator.set("x", state)
        self.acados_integrator.set("u", action_copy)
        status = self.acados_integrator.solve()
        if status != 0:
            print("Integate error!")
            exit()

        state = self.acados_integrator.get("x")
        state[9:13] /= np.linalg.norm(state[9:13])

        return state

    def step(self, action: np.ndarray):
        if self.done == 1:
            print("env is done!")
            exit()

        action_copy = action.copy()
        action_copy[action_copy>1] = 1
        action_copy[action_copy<0] = 0
        action_copy = action_copy*self.action_range[1]

        self.acados_integrator.set("x", self.state)
        self.acados_integrator.set("u", action_copy)
        status = self.acados_integrator.solve()
        if status != 0:
            print("Integate error!")
            exit()

        self.state = self.acados_integrator.get("x")
        self.state[9:13] /= np.linalg.norm(self.state[9:13])
        self.itr_num += 1
        self.reward = self.get_reward()

        if not self.is_state_available() or self.itr_num > self.max_ep_len:
            self.done = 1

        return self.state, self.reward, self.done


if __name__ == "__main__":
    env = UAVModel()
    start_time = time.time()
    for i in range(100):
        temp = env.step(np.array([6,6,6,6]))  
        env.reset()  
        # print(temp)
    end_time = time.time()
    run_time = end_time - start_time

    print("程序运行时间（秒）:", run_time)
