
import math
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import time
from random import randint
from bresenham import bresenham


class rrtstar():
    def __init__(self, obstacle_map, sx, sy, gx, gy):

        self.sx = sx
        self.sy = sy
        self.gx = gx
        self.gy = gy

        self.cvshow_ratio = 2
        self.obstacle_map_view = cv2.resize( obstacle_map.copy(), ( obstacle_map.shape[1]*self.cvshow_ratio, obstacle_map.shape[0]*self.cvshow_ratio))
        # self.obstacle_map_view = 255 - self.obstacle_map_view 
        self.obstacle_map_view[self.sy,self.sx] = [0,255,0]
        self.obstacle_map_view[self.gy,self.gx] = [0,0,255]
        cv2.circle( self.obstacle_map_view, (int(self.sx*self.cvshow_ratio) , int(self.sy*self.cvshow_ratio)), 15,(0,255,0), -1 )
        cv2.circle( self.obstacle_map_view, (int(self.gx*self.cvshow_ratio) , int(self.gy*self.cvshow_ratio)), 15,(0,0,255), -1 )
        

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
        self.min_length = 4
        self.max_length = 25
        self.goal_radius = 30

        self.reparent_radius = 110

        self.max_sample_num = 500

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
        self.but_keep_going = False
        self.found_path_once = False

        while self.reached_goal == False:
            # print('loop')

            self.num_sample_node += 1

            found_valid_point = False
            while found_valid_point == False:
                x = randint(1, self.x_size-1)
                y = randint(1, self.y_size-1)
                
                if self.obstacle_map[y,x] == 0:
                    found_valid_point = True
            # print('sample: ',x,y)

            all_dists = []
            all_total_cost = []
            all_nodes_xy = []
            for node in self.nodes :
                if (node[0] != self.gx) and ( node[1] != self.gy): 
                    d = self.dist_2_points([node[0],node[1]],[x,y])
                    all_dists.append( d )
                    all_total_cost.append(d + self.nodes[(node[0],node[1])][0])
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
                # self.nodes[(x,y)] = [ self.nodes[nearest_node][0] + all_dists[min_dist_index] , nearest_node ]
                self.nodes[(x,y)] = [ all_total_cost[min_dist_index] , nearest_node ]

                self.num_valid_node += 1

                # check if this new node is close enough to the goal, 
                # if yes, then change the FLAG, and put goal-node into the tree
                
                if self.dist_2_points( (x,y), (self.gx, self.gy) ) <= self.goal_radius :
                    self.found_path_once = True
                    self.nodes[(self.gx, self.gy)] = [ self.nodes[(x,y)][0] +self.dist_2_points( (x,y), (self.gx, self.gy) ) , (x,y) ]
                    if self.num_sample_node > self.max_sample_num:
                        self.reached_goal = True
                for nd in self.nodes:
                    nd_x, nd_y = nd[0], nd[1]
                    if (nd_x != self.sx) and ( nd_y != self.sy): 
                        dist_nd_to_xy = self.dist_2_points(nd, (x,y))
                        if dist_nd_to_xy <= self.reparent_radius:
                            current_cost = self.nodes[nd][0]
                            new_cost = self.nodes[(x,y)][0] + dist_nd_to_xy 
                            if new_cost < current_cost:
                                this_path_hit_wall = False
                                for cell in list(bresenham(x, y, nd_x, nd_y )):
                                    if self.obstacle_map[cell[1], cell[0]] != 0:
                                        this_path_hit_wall = True
                                if this_path_hit_wall == False:
                                    self.nodes[nd][1] = (x,y)
                                    self.nodes[nd][0] = new_cost

                # draw the whole tree on map
                map_for_show = self.obstacle_map_view.copy()
                
                for node in self.nodes :
                    new_p = node
                    parent_p = self.nodes[node][1]
                    cv2.line(map_for_show, self.coord_by_ratio(new_p), self.coord_by_ratio(parent_p), (0,255,0), 1 )
                    cv2.circle( map_for_show, self.coord_by_ratio(node), 6,(55,150,10), -1 )
                cv2.circle( map_for_show, self.coord_by_ratio((x,y)), 6,(0,0,250), -1 )
                
                    

                if self.found_path_once == True:
                    the_path = []
                    self.num_path_node  = 0
                    finish = False
                    the_path.append((self.gx, self.gy))
                    while finish == False:
                        # print(randint(1,10))
                        the_path.append( self.nodes[the_path[-1]][1] )
                        self.num_path_node += 1
                        # print(the_path)
                        # print((x,y),self.nodes[(x,y)][1],the_path[-1], self.nodes[the_path[-1]][1])
                        # print()
                        if the_path[-1][0] == self.sx  and the_path[-1][1] == self.sy :
                            finish = True
                        if self.num_path_node > 30:
                            finish = True
                            return
                    # print(the_path) 
                    total_cost = self.nodes[the_path[0]][0] #+ self.dist_2_points( (self.gx, self.gy), self.nodes[the_path[-1]][1] )
                    outputString = 'sample:' + str(self.num_sample_node) + ' ,valid: ' + str( self.num_valid_node) + ' , path: ' + str(self.num_path_node) + ' ,total_cost: ' + str(int(total_cost))  
                    print( outputString )

                    org = (51,62)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 0.6
                    color = (0, 0, 255)
                    thickness = 2
                    map_for_show = cv2.putText(map_for_show, outputString, org, font,  fontScale, color, thickness, cv2.LINE_AA)

                    for node in range(1,len(the_path)) :
                        new_p = the_path[node]
                        parent_p = the_path[node-1]
                        cv2.line(map_for_show, self.coord_by_ratio(new_p), self.coord_by_ratio(parent_p), (0,0,255), 7 )

                # print('now show while searching')
                self.cvshow_larger(map_for_show, 1, 50)

            # if the completion FLAG is good, then extract the best path
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
                total_cost = self.nodes[the_path[0]][0] #+ self.dist_2_points( (self.gx, self.gy), self.nodes[the_path[-1]][1] )
                outputString = 'searching...  sample:' + str(self.num_sample_node) + ' ,valid: ' + str( self.num_valid_node) + ' , path: ' + str(self.num_path_node) + ' ,total_cost: ' + str(int(total_cost))  
                print( outputString )

            

                for node in range(1,len(the_path)) :
                    new_p = the_path[node]
                    parent_p = the_path[node-1]
                    cv2.line(map_for_show, self.coord_by_ratio(new_p), self.coord_by_ratio(parent_p), (0,0,255), 7 )


                # cv2.circle( self.obstacle_map_view, (x,y), 3,(255,0,100), -1 )

                self.cvshow_larger(map_for_show, 1.0, 0)



    def coord_by_ratio(self, a):
        return (int(a[0]*self.cvshow_ratio) , int(a[1]*self.cvshow_ratio))

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
        enlarged = cv2.resize(img, (int(original_x*ratio), int(original_y*ratio)),interpolation = cv2.INTER_NEAREST)
        # print('img size', img.shape)
        # print('large size', enlarged.shape)
        cv2.imshow('plan', enlarged)
        if t != 0:
            cv2.waitKey(t)
        else:
            cv2.waitKey(0)








def main():
    # start and goal position
    
    # sx = 52
    # sy = 330
    
    # gx = 300 
    # gy = 30

    sx = 152
    sy = 39
    
    gx = 300 #313
    gy = 290 #230

    obstacle_map = cv2.imread( 'map2.bmp' )
    # obstacle_map = cv2.rotate(obstacle_map, cv2.ROTATE_90_CLOCKWISE)

    print('bmp size ', obstacle_map.shape)


    astarplanner = rrtstar(obstacle_map, sx, sy, gx, gy)



if __name__ == '__main__':
    main()
    


