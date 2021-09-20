
import math
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import time
from random import randint
from bresenham import bresenham

class rrt():
    def __init__(self, obstacle_map, sx, sy, gx, gy):

        self.show_visul_map = True

        self.sx = sx
        self.sy = sy
        self.gx = gx
        self.gy = gy


        # the array for visualization  ( not planning )
        self.obstacle_map_view = obstacle_map.copy()
        # parameters for visuliazaiton 
        self.visual_circle_radius = 7
        self.visual_circle_color = (255,50,50)
        self.visual_circle_color_start = (0,  10 , 255)
        self.visual_circle_color_goal  = (0, 255 , 10 )
        self.visual_line_explore_thick = 3
        self.visual_line_path_thick = 5
        self.visual_line_explore_color = (0,255,0)
        self.visual_line_path_color = (0,0,255)
        self.cvshow_ratio = 1
        
        self.obstacle_map_view_backup = self.obstacle_map_view.copy()

        # the array for planning  ( not visualization )
        self.obstacle_map = obstacle_map.copy()[:,:,0] 
        self.x_size = self.obstacle_map.shape[1]
        self.y_size = self.obstacle_map.shape[0]

        self.nodes = dict()  # store all nodes 

        # parameters for planning behaivor 
        self.min_length = 10
        self.max_length = 60
        self.goal_radius = 60

        self.num_sample_node = 0
        self.num_valid_node = 0
        self.num_path_node = 0

        self.reached_goal = False

        self.planning()


    def planning(self):
        self.open_nodes = dict()
        self.close_nodes = dict()

        self.nodes[(self.sx, self.sy)] = [ 0, ( self.sx, self.sy ) ]  # cost, parent_grid 

        self.reached_goal = False

        while self.reached_goal == False:

            self.num_sample_node += 1

            found_valid_point = False
            while found_valid_point == False:
                x = randint(1, self.x_size-1)
                y = randint(1, self.y_size-1)
                if self.obstacle_map[y,x] == 255:
                    found_valid_point = True
            # print('sample: ',x,y)

            all_dists = []
            all_nodes_xy = []
            for node in self.nodes :
                d = self.dist_2_points([node[0],node[1]],[x,y])
                all_dists.append(d)
                all_nodes_xy.append( ( node[0] , node[1] ) )
            all_dists_min = min(all_dists)
            all_dists = np.array( all_dists )
            min_dist_index = np.where(all_dists == all_dists_min)[0][0]
            nearest_node = all_nodes_xy[min_dist_index]
            
            # truncate the line if too long 
            discard = False
            if all_dists_min > self.min_length:
                if all_dists_min > self.max_length:
                    (x,y) = self.truncate_line( (x,y), nearest_node, all_dists_min )
            else:
                discard = True

            # check if the new line run through obstacle
            for cell in list(bresenham(x, y, nearest_node[0], nearest_node[1])):
                if self.obstacle_map[cell[1], cell[0]] == 0:
                    discard = True
            
            if discard == False:
                self.nodes[(x,y)] = [ self.nodes[nearest_node][0] + all_dists[min_dist_index] , nearest_node ]

                self.num_valid_node += 1

                if self.dist_2_points( (x,y), (self.gx, self.gy) ) <= self.goal_radius :
                    self.reached_goal = True
                    self.nodes[(self.gx, self.gy)] = [ self.nodes[(x,y)][0], (x,y) ]
                    print('Found it ')

                for node in self.nodes :
                    new_p = node
                    parent_p = self.nodes[node][1]
                    if self.show_visul_map:
                        cv2.line(self.obstacle_map_view, new_p, parent_p, self.visual_line_explore_color, self.visual_line_explore_thick )

                if self.show_visul_map:
                    cv2.circle( self.obstacle_map_view, (x,y), self.visual_circle_radius , self.visual_circle_color , -1 )
                    cv2.circle( self.obstacle_map_view, (self.sx , self.sy), self.visual_circle_radius , self.visual_circle_color_start , -1 )
                    cv2.circle( self.obstacle_map_view, (self.gx , self.gy), self.visual_circle_radius , self.visual_circle_color_goal , -1 )

                self.cvshow_larger(self.obstacle_map_view, self.cvshow_ratio, 30)

            if self.reached_goal == True:
                the_path = []
                finish = False
                the_path.append((self.gx, self.gy))
                while finish == False:
                    the_path.append( self.nodes[the_path[-1]][1] )
                    self.num_path_node += 1
                    if the_path[-1][0] == self.sx  and the_path[-1][1] == self.sy :
                        finish = True
                print(the_path) 
                print('sample, valid, path  ',self.num_sample_node, self.num_valid_node, self.num_path_node)

                total_cost = self.nodes[the_path[0]][0] #+ self.dist_2_points( (self.gx, self.gy), self.nodes[the_path[-1]][1] )
                print('total_cost ', int(total_cost))

                for node in range(1,len(the_path)) :
                    new_p = the_path[node]
                    parent_p = the_path[node-1]
                    if self.show_visul_map:
                        cv2.line(self.obstacle_map_view, new_p, parent_p, self.visual_line_path_color , self.visual_line_path_thick )

                self.cvshow_larger(self.obstacle_map_view, self.cvshow_ratio, 0)




    def dist_2_points(self, a, b):
        ax, ay = a[0], a[1]
        bx, by = b[0], b[1]
        dx = ax-bx
        dy = ay-by
        d = sqrt(dx*dx+dy*dy)
        return d

    def truncate_line( self, newpoint, nearpoint, d ):
        new_x , new_y =  newpoint[0], newpoint[1]
        near_x , near_y =  nearpoint[0], nearpoint[1]
        dy , dx = new_y - near_y  , new_x - near_x
        ratio = self.max_length / d
        new_new_x = near_x + dx * ratio
        new_new_y = near_y + dy * ratio
        return (int(new_new_x), int(new_new_y)) 


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
        if self.show_visul_map:
            cv2.imshow('plan', enlarged)
            cv2.waitKey(t)







def main():

    # start and goal position
    sx = 192
    sy = 120
    gx = 560 
    gy = 550 

    obstacle_map = cv2.imread( 'map3_large.png' )
    # obstacle_map = cv2.rotate(obstacle_map, cv2.ROTATE_90_CLOCKWISE)
    print('map size: ', obstacle_map.shape)

    astarplanner = rrt(obstacle_map, sx, sy, gx, gy)



if __name__ == '__main__':
    main()
    

