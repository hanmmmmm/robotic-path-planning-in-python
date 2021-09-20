
__author__ = "hanmmmmm"
__version__ = "1.0.0"
__maintainer__ = "hanmmmmm"

import math
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import time


'''
overall:
This is an implementation of Dijkstra path planning for 2D case 

Inputs:
- one numpy array as the map
- starting position, x,y
- goal position, x,y

Output:
self.path - a list [ (x,y) , (x2,y2) , ... ]

'''


class astar():
    def __init__(self, obstacle_map, sx, sy, gx, gy):

        self.show_visul_map = True

        self.sx = sx
        self.sy = sy
        self.gx = gx
        self.gy = gy

        self.obstacle_map_view = obstacle_map.copy()

        self.obstacle_map_view[self.sy,self.sx] = [0,255,0]
        self.obstacle_map_view[self.gy,self.gx] = [0,0,255]

        self.obstacle_map_view_backup = self.obstacle_map_view.copy()

        self.cvshow_ratio = 10

        self.path = []
        
        self.obstacle_map = obstacle_map.copy()[:,:,0] 
        
        self.x_size = self.obstacle_map.shape[1]
        self.y_size = self.obstacle_map.shape[0]

        self.open_nodes = dict()
        self.close_nodes = dict()

        self.reached_goal = False
        
        self.set_motion_model()
        
        self.planning()

        self.move_robot()


    def planning(self):
        self.open_nodes = dict()
        self.close_nodes = dict()

        self.open_nodes[(self.sx, self.sy)] = [ 0, ( self.sx, self.sy ) ]  # G,  parent_grid 
        
        self.reached_goal = False

        while self.reached_goal == False:

            ### find the OPEN grids with smallest cost 
            costs = []
            grids = []
            for i in self.open_nodes:
                grids.append(i)
                cost = self.open_nodes[i][0]
                costs.append( cost )
                
            costs_min = min(costs)
            costs = np.array( costs )
            min_costs_index = np.where(costs == costs_min)[0]

            ### iterate through the grids with smallest f-cost 
            for i in min_costs_index:    
                x = grids[i][0]
                y = grids[i][1]
                gcost = costs[i]
                l_node = self.open_nodes.pop( (x,y) )
                self.close_nodes[grids[i]] = l_node

                if x == self.gx and y == self.gy:
                    print('Found it !!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    self.reached_goal = True
                
                for nb in self.motion: # the 8 direction motions
                    # print('### i ',i,'nb ', nb)
                    newx = x + nb[0]   # x posi of one of the 8 surrounding grids
                    newy = y + nb[1]   # y posi
                    c_onestep = nb[2]  # cost from (x,y) to this surrounding grid

                     # if this grid is inside the map 
                    if 0 < newx < self.x_size and 0 < newy < self.y_size:

                        #  skip it , if it is obstiacle, or it is in CLOSE 
                        if self.obstacle_map[newy, newx] == 0 or (newx, newy) in self.close_nodes :
                            pass 

                        # otherwise, if it is NEW, then add it into OPEN, and use current grid as its parent;
                        #            if it is OPEN and there is another potential parent grid can make smaller cost, then reparent it
                        else:
                            if (newx, newy) not in self.open_nodes or ( gcost + c_onestep ) < self.open_nodes[(newx, newy)][0]:
                                self.open_nodes[(newx,newy)] = [ (gcost + c_onestep) , (x,y) ]
             
            
            # display the explored grids for visulizaiton 
            if self.show_visul_map:
                self.draw_all_nodes_for_view()
                self.cvshow_larger(self.obstacle_map_view, self.cvshow_ratio, 10)

        ###  extract path from the examined grids 
        self.path.append((self.gx, self.gy))
        while (self.sx, self.sy) not in self.path:
            x = self.path[-1][0]
            y = self.path[-1][1]
            parent_node = self.close_nodes[ (x,y) ][1]
            self.path.append(parent_node)

        self.path.reverse()
        if self.show_visul_map:
            self.draw_path_for_view()
            self.cvshow_larger(self.obstacle_map_view, self.cvshow_ratio, 0)

        print('Found the best path: ', self.path)


    def move_robot(self):
        self.robot_xy = (0,0)  # store the current position of robot 
        # self.path.append(self.path[-1])
        while self.robot_xy != (self.gx, self.gy):
            self.robot_xy = self.path[0]
            self.path.pop(0)

            self.obstacle_map_view = self.obstacle_map_view_backup.copy()
            self.obstacle_map_view[self.robot_xy[1], self.robot_xy[0]] = (0,0,255)
            self.draw_path_for_view()
            self.cvshow_larger(self.obstacle_map_view, self.cvshow_ratio, 50)




    def draw_all_nodes_for_view(self):
        '''
        Draw each point in OPEN and CLOSE groups on the map, for visulization only
        '''
        # print('open')
        for i in self.open_nodes:
            # print( i, self.open_nodes[i][0], self.open_nodes[i][1], self.open_nodes[i][2])
            self.obstacle_map_view[i[1],i[0]] = [200,100,30]
        # print('close')
        for i in self.close_nodes:
            # print(i, self.close_nodes[i][0], self.close_nodes[i][1], self.close_nodes[i][2])
            self.obstacle_map_view[i[1],i[0]] = [200,30,150]
        self.draw_ends_for_view()


    def draw_path_for_view(self):
        '''
        Draw the path points on the map, for visulization only
        '''
        for i in self.path:
            if self.obstacle_map[i[1], i[0]] != 0:
                self.obstacle_map_view[i[1],i[0]] = [0,255,30]
        self.draw_ends_for_view()


    def draw_ends_for_view(self):
        '''
        Draw the start point and finish point on the map, for visulization only
        '''
        self.obstacle_map_view[self.sy,self.sx] = [0,255,0]
        self.obstacle_map_view[self.gy,self.gx] = [0,0,255]



    def set_motion_model(self):
        # each element in the list: [ dx, dy, cost ] 
        self.motion = [[ 1,  0, 10],
                       [ 0,  1, 10],
                       [-1,  0, 10],
                       [ 0, -1, 10],
                       [-1, -1, 14],
                       [-1,  1, 14],
                       [ 1, -1, 14],
                       [ 1,  1, 14]]

    def cvshow_larger(self, img, ratio, t):
        '''
        Show the input image, with a resize ratio in the input, with waitKey time value as t
        - img: np.array, usually has shape of M x N x 3
        - ratio: float/int,  not zero
        - t: int,  unit milli-second
        '''
        original_y = img.shape[0]
        original_x = img.shape[1]
        enlarged = cv2.resize(img, (int(original_x*ratio), int(original_y*ratio)),interpolation = cv2.INTER_NEAREST)
        # print('img size', img.shape)
        # print('large size', enlarged.shape)
        cv2.imshow('plan', enlarged)
        cv2.waitKey(t) 








def main():

    # start and goal position
    sx = 16
    sy = 6
    gx = 43 #42 
    gy = 47 #37 

    obstacle_map = cv2.imread( 'map3.png' )
    # obstacle_map = cv2.rotate(obstacle_map, cv2.ROTATE_90_CLOCKWISE)
    
    print('map size: ', obstacle_map.shape)

    astarplanner = astar(obstacle_map, sx, sy, gx, gy)



if __name__ == '__main__':
    main()
    
