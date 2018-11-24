import numpy as np

class Car(object):

    action_bound = [-5, 5]
    action_dim = 2
    state_dim = (3,10)
    dt = 0.1

    def __init__(self):

        self.goal_state = [45, 6, 0, 0, 0, 0, 2, 6, 0, 0]
        self.safe_distance = 10
        self.v_max = 45
        self.lr = 2.35
        self.lf = 2.35
        self.timestep = 0.1
        self.vehicle_length = 4.7
        self.vehicle_width = 1.8


    def step(self, state, action):

        ego_cs = state[0]
        other_cs = state[1]
        obstacle_cs = state[2]

        x_cs_ego =  ego_cs[0]
        y_cs_ego =  ego_cs[1]
        vel_cs_ego = ego_cs[2]
        accel_cs_ego = ego_cs[3]
        heading_cs_ego = ego_cs[4]
        steer_angle_cs_ego = ego_cs[5]

        steer_angle_ns_ego = steer_angle_cs_ego + action[0]
        accel_ns_ego = accel_cs_ego + action[1]

        beta_cs = np.arctan(np.tan(steer_angle_cs_ego)*(self.lr/(self.lf+self.lr)))

        dxdt = vel_cs_ego + np.cos(steer_angle_cs_ego + beta_cs)
        dydt = vel_cs_ego + np.sin(steer_angle_cs_ego + beta_cs)
        dvdt = accel_ns_ego
        dthetadt = (vel_cs_ego/self.lr) * np.sin(beta_cs)

        x_ns_ego = x_cs_ego + dxdt * self.timestep
        y_ns_ego = y_cs_ego + dydt * self.timestep
        vel_ns_ego = vel_cs_ego + dvdt * self.timestep
        heading_ns_ego = heading_cs_ego + dthetadt * self.timestep

        distance_left_road_ns = 8 - y_ns_ego
        distance_right_road_ns = y_ns_ego - 0

        distance_goal_x_ns = self.goal_state[0] - x_ns_ego
        distance_goal_y_ns = self.goal_state[1] - y_ns_ego

        ego_ns = [x_ns_ego, y_ns_ego, vel_ns_ego, accel_ns_ego, heading_ns_ego, steer_angle_ns_ego, distance_left_road_ns, distance_right_road_ns, distance_goal_x_ns, distance_goal_y_ns]

        other_ns = self.other_vehicle_step(other_cs)

        obstacle_ns = obstacle_cs

        next_state = np.array([ego_ns, other_ns, obstacle_ns])

        reward, done = self.reward_function(ego_ns, other_ns, obstacle_ns)

        return next_state, reward, done


    def other_vehicle_step(self, other_cs):

        other_ns = other_cs

        if 20 <= other_cs[0] <= 21:
            other_ns[3] = np.random.choice([-2.5,2.5])

        elif 30<= other_cs[0] <= 31:
            other_ns[3] = 0       

        other_ns[2] = other_cs[2] + other_cs[3] * self.timestep
        other_ns[0] = other_cs[0] + other_cs[2] * self.timestep

        other_ns[8] = 45 - other_ns[0]

        return other_ns
    
    def reward_function(self, ego_ns, other_ns, obstacle_ns):

        reward = 0
        done = False

        distance_x_veh = np.absolute(ego_ns[0] - other_ns[0])
        distance_y_veh = np.absolute(ego_ns[1] - other_ns[1])

        distance_x_obs = np.absolute(ego_ns[0] - obstacle_ns[0])
        distance_y_obs = np.absolute(ego_ns[1] - obstacle_ns[1])

        distance_x_goal = ego_ns[8]
        distance_y_goal = ego_ns[9]

        #collision_zone

        if distance_x_veh <= self.vehicle_length and distance_y_veh <= self.vehicle_width:
            reward = reward-5
            done = True

        if distance_x_obs <= self.vehicle_length and distance_y_obs <= self.vehicle_width:
            reward = reward-5
            done = True
        
        #safe_zone_cost

        if distance_x_veh <= self.vehicle_length + 1 and distance_y_veh < self.vehicle_width +1:
            reward = reward-2

        #stays_on_road

        if not (0< ego_ns[0]< 8) or (0<ego_ns[1]<50):
            reward = reward-3
            done = True
        
        #goal-reaching 

        if distance_x_goal < 1:
            reward = reward + 5
            done  = True
        
        if distance_y_goal < 1:
            reward = reward +5
            done = True 

        #lane_changing

        if (24 <= ego_ns[0] <= 26) and (4 <= ego_ns[1] <= 8):
            reward = reward +1                  
        

        return reward, done

    
    def reset(self):

        # [x, y, vel, accel, heading,steer_angle, distance_left_road, distance_right_road, distance_goal_x, distance_goal_y]

        ego_vehicle = np.array([5,2,0,0,0,0,6,2,40,4])

        other_vehicle = np.array([5,6,0,0,0,0,2,6,40,0])

        obstacle = np.array([25,2,0,0,0,0,6,2,0,0])

        init_state = np.array([ego_vehicle, other_vehicle, obstacle])
        
        return init_state

     

    def set_fps(self, fps):
        pass

    def render(self):
        pass



    


    