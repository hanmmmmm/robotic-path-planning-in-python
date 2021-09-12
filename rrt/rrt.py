
import math
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import time
from random import randint
from bresenham import bresenham

class astar():
    def __init__(self, obstacle_map, sx, sy, gx, gy):

        self.sx = sx
        self.sy = sy
        self.gx = gx
        self.gy = gy

        self.obstacle_map_view = obstacle_map.copy()
        # self.obstacle_map_view = 255 - self.obstacle_map_view 
        self.obstacle_map_view[self.sy,self.sx] = [0,255,0]
        self.obstacle_map_view[self.gy,self.gx] = [0,0,255]
        cv2.circle( self.obstacle_map_view, (self.sx,self.sy), 6,(255,90,180), -1 )
        cv2.circle( self.obstacle_map_view, (self.gx,self.gy), 6,(255,200,50), -1 )
        self.cvshow_ratio = 2

        # self.cvshow_larger(self.obstacle_map_view, self.cvshow_ratio, 0)

        # print(self.obstacle_map_view.shape)

        
        self.obstacle_map = obstacle_map.copy()[:,:,0] 
        self.obstacle_map = 255 - self.obstacle_map 
        self.obstacle_map[self.obstacle_map == 255]  =  1
        # print(self.obstacle_map)
        # print(self.obstacle_map[230:310, 300:310])
        # self.cvshow_larger(self.obstacle_map, self.cvshow_ratio, 0)

        self.x_size = self.obstacle_map.shape[1]
        self.y_size = self.obstacle_map.shape[0]

        # self.open_nodes = dict()
        # self.close_nodes = dict()

        self.nodes = dict()
        self.min_length = 10
        self.max_length = 50
        self.goal_radius = 30

        self.num_sample_node = 0
        self.num_valid_node = 0
        self.num_path_node = 0

        self.reached_goal = False

        # self.obstacle_map[sy,sx] = 77
        # self.obstacle_map[gy,gx] = 33

        # print(self.obstacle_map)

        # self.get_motion_model()
        
        self.planning()


    def planning(self):
        self.open_nodes = dict()
        self.close_nodes = dict()



        self.nodes[(self.sx, self.sy)] = [ 0, ( self.sx, self.sy ) ]  # C, parent_grid 
        # self.open_nodes[(self.sx+2, self.sy-5)] = [ 1,1,2, [ self.sx, self.sy ] ]
        # self.open_nodes[(self.sx+1, self.sy-1)] = [ 0,0,0, [ self.sx, self.sy ] ]

        self.reached_goal = False

        while self.reached_goal == False:
            # print('')

            self.num_sample_node += 1

            found_valid_point = False
            while found_valid_point == False:
                x = randint(1, self.x_size-1)
                y = randint(1, self.y_size-1)
                if self.obstacle_map[y,x] == 0:
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
            # discard = False
            for cell in list(bresenham(x, y, nearest_node[0], nearest_node[1])):
                if self.obstacle_map[cell[1], cell[0]] != 0:
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
                    cv2.line(self.obstacle_map_view, new_p, parent_p, (0,255,0), 1 )


                cv2.circle( self.obstacle_map_view, (x,y), 3,(255,0,100), -1 )

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
                    cv2.line(self.obstacle_map_view, new_p, parent_p, (0,0,250), 1 )


                # cv2.circle( self.obstacle_map_view, (x,y), 3,(255,0,100), -1 )

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
        original_y = img.shape[0]
        original_x = img.shape[1]
        enlarged = cv2.resize(img, (original_x*ratio, original_y*ratio),interpolation = cv2.INTER_NEAREST)
        # print('img size', img.shape)
        # print('large size', enlarged.shape)
        cv2.imshow('plan', enlarged)
        if t != 0:
            cv2.waitKey(t)
        else:
            cv2.waitKey(0)








def main():
    # start and goal position
    
    sx = 152
    sy = 39
    
    gx = 300 #313
    gy = 290 #230

    obstacle_map = cv2.imread( 'map2.bmp' )
    # obstacle_map = cv2.rotate(obstacle_map, cv2.ROTATE_90_CLOCKWISE)

    print('bmp size ', obstacle_map.shape)


    astarplanner = astar(obstacle_map, sx, sy, gx, gy)



if __name__ == '__main__':
    main()
    


