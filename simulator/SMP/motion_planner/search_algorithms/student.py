from SMP.motion_planner.node import PriorityNode

from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import AStarSearch
import numpy as np

from typing import List
from commonroad.scenario.trajectory import State


class StudentMotionPlanner(AStarSearch):
    """
    Motion planner implementation by students.
    Note that you may inherit from any given motion planner as you wish, or come up with your own planner.
    Here as an example, the planner is inherited from the GreedyBestFirstSearch planner.
    """

    def __init__(self, scenario, planningProblem, automata, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automata,
                         plot_config=plot_config)

    def ego_speed(self, state_list: List[State]):
        return state_list[-1].velocity
    
    def ego_long_acc(self, state_list: List[State]):
        if (len(state_list) < 2):
            return 0
        else:
            prev = state_list[-2].velocity * np.sin(state_list[-2].orientation)
            curr = state_list[-1].velocity * np.sin(state_list[-1].orientation)
            return curr - prev
        
    def ego_lateral_acc(self, state_list: List[State]):
        if (len(state_list) <= 1):
            return 0
        else:
            prev = state_list[-2].velocity * np.cos(state_list[-2].orientation)
            curr = state_list[-1].velocity * np.cos(state_list[-1].orientation)
            return curr - prev
    def ego_long_jerk(self, state_list: List[State]):
        if (len(state_list) <= 1):
            return 0
        else:
            curr_states = state_list
            prev_states = state_list[:-1]

            acc_curr = self.ego_long_acc(curr_states)
            acc_prev = self.ego_long_acc(prev_states)

            if (acc_curr != 0 and acc_prev != 0):
                return acc_curr - acc_prev
            else:
                return 0

    def THWF(self, state_list: List[State]):
        
        if (len(state_list) <= 1):
            return 0
        else:
            curr_lane = 0
            x_dir = 0
            y_dir = 0
            obs = []

            for lanelet in self.scenario.lanelet_network.lanelets:
                if lanelet.convert_to_polygon().contains_point(state_list[-1].position):
                    curr_lane = lanelet.lanelet_id
                    obs = lanelet.get_obstacles(self.scenario.dynamic_obstacles, state_list[-1].time_step)
            pos_curr = state_list[-1].position
            pos_prev = state_list[-2].position
            
            if pos_curr[0] > pos_prev[0]:
                x_dir = 1
            else:
                x_dir = -1
            if pos_curr[1] > pos_prev[1]:
                y_dir = 1
            else:
                y_dir = -1
                
            thwf = 0
            for o in obs:
                pos_obs = o.prediction.occupancy_at_time_step(state_list[-1].time_step).shape.center
                if x_dir > 0 and pos_obs[0] > pos_curr[0] or x_dir < 0 and pos_obs[0] < pos_curr[0]:
                    if y_dir > 0 and pos_obs[1] > pos_curr[1] or y_dir < 0 and pos_obs[1] < pos_curr[1]:
                        thwf += (pos_curr[0] - pos_obs[0])/state_list[-1].velocity    
            return thwf
        
    def THWB(self, state_list: List[State]):
        if (len(state_list) <= 1):
            return 0
        else:
            curr_lane = 0
            x_dir = 0
            y_dir = 0
            obs = []

            for lanelet in self.scenario.lanelet_network.lanelets:
                if lanelet.convert_to_polygon().contains_point(state_list[-1].position):
                    curr_lane = lanelet.lanelet_id
                    obs = lanelet.get_obstacles(self.scenario.dynamic_obstacles, state_list[-1].time_step)
            pos_curr = state_list[-1].position
            pos_prev = state_list[-2].position
            
            if pos_curr[0] > pos_prev[0]:
                x_dir = 1
            else:
                x_dir = -1
            if pos_curr[1] > pos_prev[1]:
                y_dir = 1
            else:
                y_dir = -1
                
            thwb = 0
            for o in obs:
                pos_obs = o.prediction.occupancy_at_time_step(state_list[-1].time_step).shape.center
                if x_dir > 0 and pos_obs[0] < pos_curr[0] or x_dir < 0 and pos_obs[0] > pos_curr[0]:
                    if y_dir > 0 and pos_obs[1] < pos_curr[1] or y_dir < 0 and pos_obs[1] > pos_curr[1]:
                        thwb += (pos_obs[0] - pos_curr[0])/state_list[-1].velocity
                        
            return thwb
    def collision(self, state_list: List[State]):
        return 1 if self.is_collision_free(state_list) else 0
    def social_impact(self, state_list: List[State]):
        if (len(state_list) <= 1):
            return 0
        else:
            social_impact = 0
            for obs in self.scenario.dynamic_obstacles:
                v_curr = obs.prediction.trajectory.state_list[-1].velocity
                v_prev = obs.prediction.trajectory.state_list[-2].velocity
                social_impact += v_curr - v_prev if v_curr - v_prev < 0 else 0
            return social_impact

    def number_overtaken(self, state_list: List[State]):
        if (len(state_list) <= 1):
            return 0
        else:
            curr_lane = 0
            x_dir = 0
            y_dir = 0
            obs = []

            for lanelet in self.scenario.lanelet_network.lanelets:
                if lanelet.convert_to_polygon().contains_point(state_list[-1].position):
                    curr_lane = lanelet.lanelet_id
                    if (lanelet.adj_left is not None and lanelet.adj_left_same_direction):
                        left_lane = self.scenario.lanelet_network.find_lanelet_by_id(lanelet.adj_left)
                        obs += lanelet.get_obstacles(self.scenario.dynamic_obstacles, state_list[-1].time_step)
                    if (lanelet.adj_right is not None and lanelet.adj_right_same_direction):
                        left_lane = self.scenario.lanelet_network.find_lanelet_by_id(lanelet.adj_right)
                        obs += lanelet.get_obstacles(self.scenario.dynamic_obstacles, state_list[-1].time_step)
            pos_curr = state_list[-1].position
            pos_prev = state_list[-2].position
            
            if pos_curr[0] > pos_prev[0]:
                x_dir = 1
            else:
                x_dir = -1
            if pos_curr[1] > pos_prev[1]:
                y_dir = 1
            else:
                y_dir = -1
                
            n_overtaken = 0
            for o in obs:
                pos_obs = o.prediction.occupancy_at_time_step(state_list[-1].time_step).shape.center
                if x_dir > 0 and pos_obs[0] < pos_curr[0] or x_dir < 0 and pos_obs[0] > pos_curr[0]:
                    if y_dir > 0 and pos_obs[1] < pos_curr[1] or y_dir < 0 and pos_obs[1] > pos_curr[1]:
                        v_obs = o.prediction.trajectory.state_list[-1].velocity
                        if state_list[-1].velocity > v_obs:
                            n_overtaken += 1
            return n_overtaken
        
    def ttc(self, state_list: List[State]):
        if (self.ego_long_acc(state_list) < 0):
            return self.THWF(state_list)
        else:
            return 0
    def lane_change(self, state_list: List[State]):
        if (len(state_list) <= 1):
            return 0
        else:
            curr_lane = 0
                
            for lanelet in self.scenario.lanelet_network.lanelets:
                if lanelet.convert_to_polygon().contains_point(state_list[-1].position):
                    curr_lane = lanelet.lanelet_id
            for state in state_list[:-1]:
                for lanelet in self.scenario.lanelet_network.lanelets:
                    if lanelet.convert_to_polygon().contains_point(state.position) and lanelet.lanelet_id != curr_lane:
                        return 1
            return 0
            
    def heuristic_function(self, node_current: PriorityNode) -> float:
        state_list = node_current.list_paths[-1]
        good_theta = np.array([4.70769403, 0.68082162, -0.51429344, -0.58462425, -6.37842922, 4.14973407, -3.20588083, -4.15468459, -0.51645378, -3.23539758, 0.0721531])
        bad_theta = np.array([4.18173671, -2.82018192, -3.35219218, -0.74862845, -0.63346181, 0.256896, -2.82336895, -0.86306923, -0.39304175, 0.19994144, -0.28608076])
        
        human_eval = 0
        if state_list is not None:
            ego_speed = self.ego_speed(state_list)
            ego_longitudinal_acc = self.ego_long_acc(state_list)
            ego_lateral_acc = self.ego_lateral_acc(state_list)
            ego_longitudial_jerk = self.ego_long_jerk(state_list)
            THWF = 0
            try: 
                THWF = self.THWF(state_list)
            except:
                print('error on THWF')
            THWB = 0
            try: 
                THWB = self.THWB(state_list)
            except:
                print('error on THWB')
            collision = self.collision(state_list)
            social_impact = 0
            try:
                social_impact = self.social_impact(state_list)
            except:
                print('attribute error on social impact')
            social_impact = self.social_impact(state_list)
            ttc = 0
            try:
                ttc = self.ttc(state_list)
            except:
                print('attribute error on ttc')
            num_overtaken = 0
            try:
                num_overtaken = self.number_overtaken(state_list)
            except:
                print('attribute error on num_overtaken')
            lane_changed = 0
            try:
                lane_changed = self.lane_change(state_list)
            except:
                print('attribute error on lane_changed')
            features = np.array([ego_speed, abs(ego_longitudinal_acc), abs(ego_lateral_acc), abs(ego_longitudial_jerk), THWF, THWB, collision, social_impact, ttc, num_overtaken, lane_changed])
            
            human_eval = np.dot(good_theta, features)
       
        if self.reached_goal(node_current.list_paths[-1]):
            return 0.0 + human_eval

        if self.position_desired is None:
            return self.time_desired.start - node_current.list_paths[-1][-1].time_step + human_eval

        else:
            velocity = node_current.list_paths[-1][-1].velocity

            if np.isclose(velocity, 0):
                return np.inf

            else:
                return (self.calc_euclidean_distance(current_node=node_current) / velocity) + human_eval

        
