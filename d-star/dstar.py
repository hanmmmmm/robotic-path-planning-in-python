
import math
from math import cos, sqrt
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import time



class dstar():
    def __init__(self, obstacle_map, sx, sy, gx, gy):

        self.sx = gx   # start position and end position are inverted for D*
        self.sy = gy
        self.gx = sx
        self.gy = sy

        self.obstacle_map_view = obstacle_map.copy()
        # self.obstacle_map_view = 255 - self.obstacle_map_view 
        self.obstacle_map_view[self.sy,self.sx] = [0,255,0]
        self.obstacle_map_view[self.gy,self.gx] = [0,0,255]
        # self.obstacle_map_view[self.obstacle_map_view > 25]  =  255
        self.obstacle_map_view_backup = self.obstacle_map_view.copy()
        self.cvshow_ratio = 16

        
        self.obstacle_map = obstacle_map.copy()[:,:,0] 
        # self.obstacle_map = 255 - self.obstacle_map 
        # self.obstacle_map[self.obstacle_map == 255]  =  1
        # print(self.obstacle_map)
        # print(self.obstacle_map[0:10, 0:20])
        # self.cvshow_larger(self.obstacle_map, self.cvshow_ratio, 0)

        self.x_size = self.obstacle_map.shape[1]
        self.y_size = self.obstacle_map.shape[0]

        self.open_nodes = dict()
        self.close_nodes = dict()

        self.first_path_found = False

        self.clear_to_move, self.cannot_moveto = [], [] 
        
        self.set_motion_model()

        self.kmin = 0
        self.dummy_cost = 999999
        
        # self.Dijkstra_planning()  # initial planning, on the static map 

        self.open_nodes[(self.sx, self.sy)] = [ 0, 0, (self.sx,self.sy) ]  # G, k, parent_grid 

        while self.first_path_found == False:
            self.process_state()
            self.draw_all_nodes_for_view()
            self.cvshow_larger(self.obstacle_map_view, self.cvshow_ratio, 10)
            
        
        print('OPEN nodes number: ',len(self.open_nodes))
        print('CLOSE nodes number: ',len(self.close_nodes))
        print()

        ### extract the path, starting from the 'goal' 
        self.path = []
        self.path.append([self.gx, self.gy])
        while (self.sx, self.sy) not in self.path:
            x = self.path[-1][0] # the last grid in the current portion of extracted path 
            y = self.path[-1][1]
            parent_node = self.close_nodes[ (x,y) ][2] # get its parent grid
            self.path.append(parent_node)              # add this parent grid into the path
        ###  show the extracted path 
        self.draw_path_for_view()
        self.cvshow_larger(self.obstacle_map_view, self.cvshow_ratio, 200)
        print('Found the initial path, ready to move')
        print('path: ', self.path)


        self.reach_goal = False
        self.rbt_x, self.rbt_y = -1, -1 
        
        step = 0
        print('robot starts moving ')
        while self.reach_goal == False:
            
            ###  add obstacles onto map
            if step == 5:
                new_obs = [[20,17],[22,25]]#
                self.update_obstacle_map( new_obs, [])
            if step == 15:
                new_obs = [[25,22],[26,45]]#
                self.update_obstacle_map( new_obs, [])
            if step == 48:
                new_obs = [[34,29],[36,39]]#
                self.update_obstacle_map( new_obs, [])
            if step == 53:
                new_obs = [[40,20],[42,29]]#
                self.update_obstacle_map( new_obs, [])
            if step == 62:
                new_obs = [[41,33],[51,34]]#
                self.update_obstacle_map( new_obs, [])

                new_obs = [[44,38],[54,39]]#
                self.update_obstacle_map( new_obs, [])


            ##### move robot or find path

            # get a clean copy of the map for visulization 
            self.obstacle_map_view = self.obstacle_map_view_backup.copy()

            # move robot 
            self.rbt_x = self.path[0][0]
            self.rbt_y = self.path[0][1]
            self.obstacle_map_view[self.rbt_y,self.rbt_x] = [255,255,255]
            
            print('moved one step, now at ',self.rbt_x, self.rbt_y)
            
            # if robot position is identicle to the goal position
            if self.rbt_x == self.sx and self.rbt_y == self.sy:
                print('robot reached goal')
                self.reach_goal = True
            
            # if robot position is NOT identicle to the goal position
            else:
                self.rbt_next_x = self.path[1][0]
                self.rbt_next_y = self.path[1][1]

                # if the next step is clear to move
                if self.obstacle_map[self.rbt_next_y, self.rbt_next_x] == 255:
                    # print('next grid clear')
                    if len(self.path) >= 1:
                        self.path.pop(0)
                        self.draw_path_for_view()
                        step += 1
                    else:
                        self.reach_goal = True
                
                # if the next step is obstacle now
                else:
                    print('Next grid blocked, looking for new path --------------')
                    
                    # found_new_path = False
                    temp_path = [(self.rbt_x, self.rbt_y)]
                    v_rbt_x, v_rbt_y = self.rbt_x, self.rbt_y
                    v_rbt_step = 0
                    
                    while (self.sx, self.sy) not in temp_path:
                        # print('goal ', (self.sx, self.sy), (self.sx, self.sy) in temp_path )
                        
                        (v_rbt_x, v_rbt_y) = temp_path[v_rbt_step]
                        print('\nRobot at', (self.rbt_x, self.rbt_y) ,'. VRobot at', (v_rbt_x, v_rbt_y), '. Temp_path', temp_path)
                        
                        (v_rbt_x, v_rbt_y) = temp_path[v_rbt_step]
                        
                        stateh, statek, stateparent = self.get_h_k_parent((v_rbt_x, v_rbt_y))
                        
                        self.modify_cost((v_rbt_x, v_rbt_y))
                        
                        self.process_state()
                        stateh, statek, stateparent = self.get_h_k_parent((v_rbt_x, v_rbt_y))
                        
                        if stateparent != temp_path[-1]:
                            temp_path.append(stateparent)
                        
                        v_rbt_step += 1

                    self.path = temp_path
                    # print('new path to goal: ', self.path)

                    self.draw_path_for_view()
                    self.cvshow_larger(self.obstacle_map_view, self.cvshow_ratio, 50)

                self.obstacle_map_view[self.rbt_y,self.rbt_x] = [0,0,255]
                self.cvshow_larger(self.obstacle_map_view, self.cvshow_ratio, 50)
            


    def get_h_k_parent(self, grid):
        '''
        - grid: (x,y)
        Returns the h-cost, key-value, and parent grid xy of the input grid
        '''

        if grid in self.open_nodes and grid not in self.close_nodes:
            return self.open_nodes[grid]

        if grid in self.close_nodes and grid not in self.open_nodes:
            return self.close_nodes[grid]

        if grid in self.close_nodes and grid in self.open_nodes:
            print('BOTHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')
            return('check this grid, it is in both open and close')

        if grid not in self.close_nodes and grid not in self.open_nodes:
            print('NONEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE')
            return('check this grid, it seems to be new')




    def process_state(self):
        '''
        This is the PROCESS-STATE function from the D* paper
        '''
        
        ### get k_min
        ks = []
        for i in self.open_nodes:
            ks.append( self.open_nodes[i][1] )
        self.kmin = min(ks)
        # print('kmin ', self.kmin )
        
        ### get grids with the kmin
        Xs = []
        for i in self.open_nodes:
            if self.open_nodes[i][1] == self.kmin:
                Xs.append(i)
                # print(i, self.open_nodes[i])
        # print('Xs, open with kmin: ', Xs)
        if Xs == []:
            return -1

        ### get k_old
        self.kold = self.kmin

        ### iterate through all the grids have k=kmin
        for grid in Xs:

            # move grid from OPEN into CLOSED 
            self.close_nodes[grid] = self.open_nodes.pop(grid)

            x , y = grid[0], grid[1]
            cost = self.close_nodes[grid][0]

            # The first type in process-state 
            if self.kold < cost:
                # print('case 1 ')
                # found_a_new_parent = False
                best_nb = [-1, -1, 999999]

                for nb in self.motion: # the 8 direction motions
                    newx = x + nb[0]   # x posi of one of the 8 surrounding grids
                    newy = y + nb[1]   # y posi
                    c_onestep = nb[2]  # cost from (x,y) to this surrounding grid

                    if 0 < newx < self.x_size and 0 < newy < self.y_size and self.obstacle_map[newy,newx] != 0: # if this grid is inside the map and not obstacle 
                        if (newx, newy) in self.open_nodes:
                            nb_original_cost = self.open_nodes[(newx, newy)][0]
                        elif (newx, newy) in self.close_nodes:
                            nb_original_cost = self.close_nodes[(newx, newy)][0]

                        if nb_original_cost + c_onestep < best_nb[-1]:
                            best_nb = [newx, newy, nb_original_cost + c_onestep]

                        if ( nb_original_cost <= self.kold + 6 ) and (cost > (nb_original_cost + c_onestep) ):
                            self.close_nodes[grid][2] = (best_nb[0], best_nb[1])
                            self.close_nodes[grid][0] = best_nb[2]
                            # found_a_new_parent = True


                # if found_a_new_parent == False:
                #     raise ValueError('It\'s a trap !!!!!!!!!!!!!!!!!!!!!!!')
                    

            ## The first type in process-state 
            elif self.kold == cost:
                # print('case 2 ')
                for nb in self.motion: # the 8 direction motions
                    newx = x + nb[0]   # x posi of one of the 8 surrounding grids
                    newy = y + nb[1]   # y posi
                    c_onestep = nb[2]  # cost from (x,y) to this surrounding grid

                    if 0 < newx < self.x_size and 0 < newy < self.y_size and self.obstacle_map[newy,newx] != 0: # if this grid is inside the map
                        if ((newx, newy) not in self.open_nodes) and ((newx, newy)  not in self.close_nodes):
                            nb_tag = 'new'
                        if (newx, newy) in self.open_nodes:
                            nb_tag = 'open'
                            nb_parent = self.open_nodes[(newx, newy)][2]
                            nb_original_cost = self.open_nodes[(newx, newy)][0]
                        if (newx, newy) in self.close_nodes:
                            nb_tag = 'close'
                            nb_parent = self.close_nodes[(newx, newy)][2]
                            nb_original_cost = self.close_nodes[(newx, newy)][0]
                        if nb_tag == 'new' or (nb_parent==grid and nb_original_cost != (cost+c_onestep)) or (nb_parent!=grid and nb_original_cost > (cost+c_onestep)):
                            self.open_nodes[(newx,newy)] = [ 0, 0, (0,0) ]  # G, k, parent_grid
                            self.open_nodes[(newx,newy)][2] = grid
                            self.insert( (newx,newy), nb_tag, cost+c_onestep )


        ###  get kmin 
        ks = []
        for i in self.open_nodes:
            ks.append( self.open_nodes[i][1] )
        self.kmin = min(ks)


        ###  check if goal is inside of CLOSE
        if (self.gx, self.gy) in self.close_nodes:
            self.first_path_found = True
        


    def insert(self, grid, tag, hnew ):
        '''
        This is the INSERT function from the D* paper
        - grid: (x,y)
        - tag: string 'new'  'open'   'close'
        - hnew: float number 
        The h-cost of this grid is updated
        '''
        if tag == 'new':
            self.open_nodes[grid][1] = hnew
        elif tag == 'open':
            self.open_nodes[grid][1] = min(self.open_nodes[grid][1], hnew)
        elif tag == 'close':
            self.open_nodes[grid][1] = min(self.close_nodes[grid][0], hnew)
        self.open_nodes[grid][0] = hnew
        if grid in self.close_nodes:
            self.close_nodes.pop(grid)
            
    def modify_cost(self,grid):
        '''
        This is the MODIFT-COST  function from the D* paper
        - grid: (x,y)
        The h-cost of this grid is updated to the dummy large value
        '''
        hnew = self.dummy_cost
        if grid in self.close_nodes:
            self.open_nodes[grid] = self.close_nodes.pop(grid)
            self.open_nodes[grid][1] = min(self.open_nodes[grid][0], hnew)
        elif grid in self.open_nodes:
            self.open_nodes[grid][1] = min(self.open_nodes[grid][0], hnew)
        self.open_nodes[grid][0] = hnew


    # def Dijkstra_planning(self):

    #     self.open_nodes[(self.sx, self.sy)] = [ 0, 0, (self.sx,self.sy) ]  # G, k, parent_grid 

    #     self.first_path_found = False

    #     while self.first_path_found == False:
    #         # print('')
    #         gcosts = []
    #         grids = []
    #         for i in self.open_nodes:
    #             grids.append(i)
    #             gcosts.append( self.open_nodes[i][0] )
    #         gcosts_min = min(gcosts)
    #         gcosts = np.array( gcosts )
    #         min_gcosts_index = np.where(gcosts == gcosts_min)[0]

    #         ### expand on the grids which have the minimum cost
    #         for i in min_gcosts_index:    
    #             # get its info
    #             x = grids[i][0]
    #             y = grids[i][1]
    #             gcost = gcosts[i]
    #             # move it into CLOSE 
    #             self.close_nodes[grids[i]] = self.open_nodes.pop( (x,y) )

    #             # check if it reaches the goal 
    #             if x == self.gx and y == self.gy:
    #                 print('Found it !!!!!!!!!!!!!!!!!!!!!!!!!!!')
    #                 self.first_path_found = True
                
    #             # expansion in its surrounding grids
    #             for nb in self.motion: # the 8 direction motions
    #                 newx = x + nb[0]   # x posi of one of the 8 surrounding grids
    #                 newy = y + nb[1]   # y posi
    #                 c_onestep = nb[2]  # cost from (x,y) to this surrounding grid
    #                 new_cost = gcost + c_onestep   # total cost for this surrounding grid

    #                 if 0 < newx < self.x_size and 0 < newy < self.y_size: # if this grid is inside the map
    #                     if self.obstacle_map[newy, newx] == 0 or (newx, newy) in self.close_nodes : # if this grid is a obstical or in CLOSE_nodes
    #                         pass  
    #                     else: # if this grid is in NEW or OPEN_nodes, then add this grid in OPEN (or update values) 
    #                         if (newx, newy) not in self.open_nodes or new_cost < self.open_nodes[(newx, newy)][0]: 
    #                             self.open_nodes[ (newx, newy) ] = [ new_cost , new_cost , [x,y] ] 
                            
    #         # update map for visul 
    #         self.draw_all_nodes_for_view()
    #         self.cvshow_larger(self.obstacle_map_view, self.cvshow_ratio, 50)


    #     ### extract the path, starting from the 'goal' 
    #     self.path = []
    #     self.path.append([self.gx, self.gy])
        
    #     backed = False
    #     while backed == False:
    #         x = self.path[-1][0] # the last grid in the current portion of extracted path 
    #         y = self.path[-1][1]
    #         parent_node = self.close_nodes[ (x,y) ][2] # get its parent grid
    #         self.path.append(parent_node)              # add this parent grid into the path
    #         if self.sx == parent_node[0] and self.sy == parent_node[1]:  # check if this reach the end
    #             backed = True                                            # if yes, then exit this while loop
        

    #     ###  show the extracted path 
    #     self.draw_path_for_view()
    #     self.cvshow_larger(self.obstacle_map_view, self.cvshow_ratio, 0)


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

    def update_obstacle_map(self, new_obs, cleared):
        '''
        update the map array, according to the inputs:
        - new_obs: [ [a,b], [c,d] ]  or []
        - cleared: [ [a,b], [c,d] ]  or []

        Inputs are the corners of rectangles:

        [a,b]-----------
        ----------------
        ----------------
        -----------[c,d]

        This func also upadte the affected grid state, like OPEN, CLOSE, NEW 
        '''

        if len(new_obs)!=0:
            # print('add obstacle to map')
            x1 = new_obs[0][0]
            y1 = new_obs[0][1]
            x2 = new_obs[1][0]
            y2 = new_obs[1][1]
            for x in list(range(x1,x2)):
                for y in list(range(y1, y2)):
                    print(x,y)
                    self.obstacle_map[ y, x ] = 0
                    self.obstacle_map_view_backup[ y, x ] = (0,0,0)
                    if (x,y) in self.close_nodes:
                        self.close_nodes.pop((x,y))
                    if (x,y) in self.open_nodes:
                        self.open_nodes.pop((x,y))

        if len(cleared)!=0:
            # print('remove obstacle from map')
            for i in cleared:
                self.obstacle_map[ i[1], i[0] ] = 255
                self.obstacle_map_view_backup[ i[1], i[0] ] = [255,255,255]






def main():

    # start and goal position
    sx = 16
    sy = 6
    gx = 43 #42 
    gy = 47 #37 

    obstacle_map = cv2.imread( 'map3.png' )
    # obstacle_map = cv2.rotate(obstacle_map, cv2.ROTATE_90_CLOCKWISE)
    
    print('map size: ', obstacle_map.shape)

    astarplanner = dstar(obstacle_map, sx, sy, gx, gy)



if __name__ == '__main__':
    main()
    


