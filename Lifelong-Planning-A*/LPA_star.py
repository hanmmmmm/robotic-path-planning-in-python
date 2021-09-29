
import math
from math import cos, sqrt, inf
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import time



class lpastar():
    def __init__(self, obstacle_map, sx, sy, gx, gy):

        self.show_visul_map = True

        self.sx = sx   # start position and end position  
        self.sy = sy
        self.gx = gx
        self.gy = gy

        self.rbt_x, self.rbt_y = self.sx, self.sy

        self.goal_grid = (self.gx, self.gy)

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

        # self.open_nodes = dict()
        # self.close_nodes = dict()
        self.queue = dict()
        self.nonqueue = dict()

        for i in range(self.y_size):
            for j in range(self.x_size):
                if self.obstacle_map[j,i] != 0:
                    self.nonqueue[(i, j)] = [inf, inf, self.calc_h_cost((i, j),(self.gx, self.gy)), inf, inf, (inf, inf)]  # g, rhs, h, k1, k2, (x_p, y_p)

        # print(self.nonqueue)
        # print(self.queue)

        self.nonqueue[(self.sx, self.sy)][1] = 0
        self.queue[(self.sx, self.sy)] = self.nonqueue.pop((self.sx, self.sy))

        self.queue[(self.sx, self.sy)][3] = self.queue[(self.sx, self.sy)][2]
        self.queue[(self.sx, self.sy)][4] = 0

        # print(self.nonqueue)
        print(self.queue)

        self.first_path_found = False


        self.set_motion_model()

        self.get_short_path()
        self.extract_path()

        self.move_robot()


    def get_short_path(self):
        print('\nrunning: get_short_path() ')
        ite = 0
        while True:
            topkey,  topkey_grids = self.topkey()
            
            ite += 1
            print('\nite: ',ite)
            print('topkey: ', topkey)
            print('topkey_grids: ', topkey_grids)

            # print('queue:')
            # for q in self.queue:
            #     print(q, self.queue[q])

            g_k1 , g_k2 = self.calc_k(self.goal_grid)
            g, rhs, h = self.get_grid_content(self.goal_grid)
            if (topkey[0]<g_k1 or (topkey[0]==g_k1 and topkey[1]<g_k2) ) or (rhs != g):
                for u in topkey_grids:
                    self.nonqueue[u] = self.queue.pop(u)
                    g, rhs, h = self.get_grid_content(u)
                    # print('just poped ',g, rhs, h)
                    if g > rhs:
                        # print('g > rhs')
                        g = rhs
                        self.nonqueue[u][0] = g
                        
                        g, rhs, h = self.get_grid_content(u)
                        # print('updated g :',g, rhs, h)

                        x , y = u[0], u[1]
                        for nb in self.motion: # the 8 direction motions
                            newx = x + nb[0]   # x posi of one of the 8 surrounding grids
                            newy = y + nb[1]   # y 
                            if 0 < newx < self.x_size and 0 < newy < self.y_size and self.obstacle_map[newy,newx] != 0:
                                # print('\nnb of topkey_grid : ',(newx,newy))
                                # c_onestep = nb[2]  # cost from (x,y) to this surrounding grid
                                self.update_vertex((newx, newy))

                    else:
                        g = inf
                        self.nonqueue[u][0] = g
                        g, rhs, h = self.get_grid_content(u)
                        # print('updated g :',g, rhs, h)

                        x , y = u[0], u[1]
                        for nb in self.motion: # the 8 direction motions
                            newx = x + nb[0]   # x posi of one of the 8 surrounding grids
                            newy = y + nb[1]   # y 
                            if 0 < newx < self.x_size and 0 < newy < self.y_size and self.obstacle_map[newy,newx] != 0:
                                # print('\nnb of topkey_grid : ',(newx,newy))
                                # c_onestep = nb[2]  # cost from (x,y) to this surrounding grid
                                self.update_vertex((newx, newy))
                        self.update_vertex((x,y))



            else:
                print('reached-------------')
                break

            self.draw_all_nodes_for_view()
            self.cvshow_larger(self.obstacle_map_view, self.cvshow_ratio, 5)
        
        print('done')

    def topkey(self):
        # print('=======================')
        grids = []
        k1 = []
        k2 = []
        for i in self.queue:
            grids.append( i )
            k1.append( self.queue[i][3] )
            k2.append( self.queue[i][4] )

        small_k1 = min(k1)

        # print('grids', grids, 'k1',k1, 'k2',k2, 'small_k1', small_k1)

        small_k1_index = []
        for i in range(len(self.queue)):
            if k1[i] == small_k1:
                small_k1_index.append(i)
        
        k2s = []
        for i in small_k1_index:
            # print('i', i, ' in ', small_k1_index)
            k2s.append(k2[i])
        small_k2 = min(k2s)

        topkey = (small_k1, small_k2)

        topkey_grids = []
        for i in small_k1_index:
            if k2[i] == small_k2:
                topkey_grids.append(grids[i])

        # print('\ngrids', grids)
        # print('k1',k1 )
        # print('k2',k2 )
        # print('small_k1', small_k1, 'small_k2', small_k2)
        # print('small_k1_index', small_k1_index, 'small_k2', small_k2)
        
        # print(small_k1, small_k2)
        # print('=======================')

        return topkey,  topkey_grids






    def calc_k(self, grid):
        if grid in self.queue:
            [g, rhs, h, *rest] = self.queue[grid]
        elif grid in self.nonqueue:
            [g, rhs, h, *rest] = self.nonqueue[grid]
        
        k1 = min(g, rhs)  + h
        k2 = min(g, rhs)

        return k1, k2


    def get_grid_content(self, grid):
        print('get_grid_content ', grid )
        if grid in self.queue:
            [g, rhs, h, *rest] = self.queue[grid]
        elif grid in self.nonqueue:
            [g, rhs, h, *rest] = self.nonqueue[grid]
        
        rhs = self.calc_rhs(grid)[0]
        return g, rhs, h


    def calc_rhs(self, grid ):
        # print('grid ',grid)
        if grid == (self.sx, self.sy):
            return 0, 0
        else:
            x, y = grid[0], grid[1]
            all8rhs = []
            nb_xys = []
            found_robot = False
            for nb in self.motion: # the 8 direction motions
                newx = x + nb[0]   # x posi of one of the 8 surrounding grids
                newy = y + nb[1]   # y posi
                c_onestep = nb[2]  # cost from (x,y) to this surrounding grid
                
                if 0 < newx < self.x_size and 0 < newy < self.y_size and self.obstacle_map[newy,newx] != 0: # if this grid is inside the map and not obstacle 
                    # print('\nfinding rhs: nb of active grid : ',(newx,newy))
                    # g, rhs, h = self.get_grid_content((newx, newy))
                    newgrid = (newx, newy)



                    newgrid_parent = self.get_parent(newgrid)
                    if newgrid_parent != (x,y):




                        if newgrid in self.queue:
                            [g, rhs, h, *rest] = self.queue[newgrid]
                        elif newgrid in self.nonqueue:
                            [g, rhs, h, *rest] = self.nonqueue[newgrid]
                        rhs_candi = g + c_onestep
                        # print('g + c_onestep: ', new_g)
                        all8rhs.append(rhs_candi)  
                        nb_xys.append((newx, newy))

                        if (newx, newy) == (self.rbt_x, self.rbt_y):
                            best_nb = (newx, newy)
                            found_robot = True
                    
            if len(all8rhs) == 0:
                self.nonqueue[grid] = self.queue.pop(grid)
            else:

                # print('all potential rhs: ', gs)
                rhs = min(all8rhs)
                # if rhs != inf:
                best_index = all8rhs.index(rhs)
                if found_robot == False:
                    best_nb = nb_xys[best_index]
                return rhs, best_nb


    def update_vertex(self, grid):
        if self.obstacle_map[grid[1], grid[0]] != 0:
            rhs, nb = self.calc_rhs(grid)
            print('running: update_vertex',grid,'   new rhs:', rhs, '  best nb: ', nb)

            if grid in self.nonqueue:
                parent_node = self.nonqueue[ grid ][5] # get its parent grid
            if grid in self.queue:
                parent_node = self.queue[ grid ][5] # get its parent grid

            if parent_node != nb:
                if grid in self.nonqueue:
                    self.nonqueue[ grid ][5] = nb # get its parent grid
                if grid in self.queue:
                    self.queue[ grid ][5] = nb # get its parent grid
            
            
            if grid != (self.sx, self.sy):
                print('update vertex, if 1 true')
                # rhs = self.calc_rhs(grid)
                if grid in self.queue:
                    self.queue[grid][1] = rhs
                elif grid in self.nonqueue:
                    self.nonqueue[grid][1] = rhs
            if grid in self.queue:
                print('update vertex, if 2 true')
                self.nonqueue[grid] = self.queue.pop(grid)
            g, old_rhs, h = self.get_grid_content(grid)
            if g != rhs :
                print('update vertex, if 3 true')
                k1, k2 = self.calc_k(grid)
                self.insert(grid, k1, k2, nb)



    def insert(self, grid, k1, k2, nb ):
        if grid in self.nonqueue:
            self.queue[grid] = self.nonqueue.pop(grid)
            self.queue[grid][1] = self.calc_rhs(grid)[0]
            self.queue[grid][3] = k1
            self.queue[grid][4] = k2
            self.queue[grid][5] = nb
            
        elif grid in self.queue:
            self.queue[grid][1] = self.calc_rhs(grid)[0]
            self.queue[grid][3] = k1
            self.queue[grid][4] = k2
            self.queue[grid][5] = nb



    def move_robot(self):

        self.reach_goal = False
        # self.rbt_x, self.rbt_y = -1, -1 
        
        step = 0
        print('robot starts moving ')
        while self.reach_goal == False:
            
            ###  add obstacles onto map
            if step == 5:
                # new_obs = [[20,17],[22,25]]#
                new_obs = [[21,20],[22,21]]  #
                self.update_obstacle_map( new_obs, [])
            if step == 11:
                new_obs = [[21,30],[27,31]]  #
                self.update_obstacle_map( new_obs, [])
            # if step == 28:
            #     new_obs = [[34,29],[36,39]]#
            #     self.update_obstacle_map( new_obs, [])
            # if step == 33:
            #     new_obs = [[40,20],[42,29]]#
            #     self.update_obstacle_map( new_obs, [])
            if step == 22:
                # new_obs = [[41,33],[51,34]]#
                new_obs = [[41,30],[43,31]]
                self.update_obstacle_map( new_obs, [])

                # new_obs = [[44,38],[54,39]]#
                # self.update_obstacle_map( new_obs, [])


            ##### move robot or find path

            # get a clean copy of the map for visulization 
            self.obstacle_map_view = self.obstacle_map_view_backup.copy()

            # move robot 
            self.rbt_x = self.path[0][0]
            self.rbt_y = self.path[0][1]
            self.obstacle_map_view[self.rbt_y,self.rbt_x] = [255,255,255]
            
            print('moved one step, now at ',self.rbt_x, self.rbt_y)
            
            # if robot position is identicle to the goal position
            if self.rbt_x == self.gx and self.rbt_y == self.gy:
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

                    # next_grid = (self.rbt_next_x, self.rbt_next_y)
                    # x, y = next_grid[0], next_grid[1]
                    # # all8rhs = [] 
                    # # nb_xys = []  
                    # for nb in self.motion: # the 8 direction motions
                    #     newx = x + nb[0]   # x posi of one of the 8 surrounding grids
                    #     newy = y + nb[1]   # y posi
                    #     c_onestep = nb[2]  # cost from (x,y) to this surrounding grid
                        
                    #     if 0 < newx < self.x_size and 0 < newy < self.y_size and self.obstacle_map[newy,newx] != 0: # if this grid is inside the map and not obstacle 
                    #         # print('\nfinding rhs: nb of active grid : ',(newx,newy))
                    #         # g, rhs, h = self.get_grid_content((newx, newy))
                    #         newgrid = (newx, newy)
                    #         print('\ncheck new grid ', newgrid)
                    #         self.update_vertex((newx, newy))

                    #         self.draw_all_nodes_for_view()
                    #         self.cvshow_larger(self.obstacle_map_view, self.cvshow_ratio, 10)

                    self.get_short_path()
                    g, old_rhs, h = self.get_grid_content((26,25))
                    print('eeeeeeeeeee ', g, old_rhs, h )
                    self.extract_path()
                    

                self.obstacle_map_view[self.rbt_y,self.rbt_x] = [0,0,255]
                self.cvshow_larger(self.obstacle_map_view, self.cvshow_ratio, 150)


    def get_parent(self, grid):
        if grid in self.nonqueue:
            parent_node = self.nonqueue[ grid ][5] # get its parent grid
        if grid in self.queue:
            parent_node = self.queue[ grid ][5] # get its parent grid

        return parent_node

    def extract_path(self):
        print('running: extract_path()')
        self.path = []
        self.path.append([self.gx, self.gy])
        while (self.rbt_x, self.rbt_y) not in self.path:# or len(self.path) < 20:
            x = self.path[0][0] # the last grid in the current portion of extracted path 
            y = self.path[0][1]
            if (x,y) in self.nonqueue:
                parent_node = self.nonqueue[ (x,y) ][5] # get its parent grid
            if (x,y) in self.queue:
                parent_node = self.queue[ (x,y) ][5] # get its parent grid
            print('last grid: ', (x,y), ' parent: ', parent_node)
            if parent_node == (x,y):
                print('parent_node == (x,y) !!!!!!!!!!')
                break
            self.path.insert(0, parent_node)              # add this parent grid into the path
            
        ###  show the extracted path 
        if self.show_visul_map:
            self.draw_path_for_view()
            self.cvshow_larger(self.obstacle_map_view, self.cvshow_ratio, 0)
        print('Found the initial path, ready to move')
        print('path: ', self.path)

            
    # def modify_cost(self,grid):
    #     '''
    #     This is the MODIFT-COST  function from the D* paper
    #     - grid: (x,y)
    #     The h-cost of this grid is updated to the dummy large value
    #     '''
    #     hnew = self.dummy_cost
    #     if grid in self.close_nodes:
    #         self.open_nodes[grid] = self.close_nodes.pop(grid)
    #         self.open_nodes[grid][1] = min(self.open_nodes[grid][0], hnew)
    #     elif grid in self.open_nodes:
    #         self.open_nodes[grid][1] = min(self.open_nodes[grid][0], hnew)
    #     self.open_nodes[grid][0] = hnew


    def calc_h_cost(self, a, b):
        '''
        computes the H-cost between two grids a and b
        - a: (x,y)
        - b: (x,y)
        returns the h-cost as a number 
        '''
        
        ax = a[0]
        ay = a[1]
        
        bx = b[0]
        by = b[1]

        # print('calc_h_cost ', ax,ay,bx,by)
        
        if ax == bx  and ay != by:
            diag_part = 0
            line_part = abs(ay-by) *10
        elif ax != bx  and ay == by:
            diag_part = 0
            line_part = abs(ax-bx) *10
        elif ax != bx and ay != by:
            dx = abs(ax-bx)
            dy = abs(ay-by)
            if dx >= dy:
                diag_part = dy * 14
                line_part = (dx - dy) * 10
            if dy > dx:
                diag_part = dx * 14
                line_part = (dy - dx) * 10 
        
        if ax == bx  and ay == by:
            hcost = 0
        else: 
            hcost = diag_part + line_part

        # dx = abs(ax-bx)
        # dy = abs(ay-by)
        # hcost = (dx + dy) * 10

        # dx = abs(ax-bx)
        # dy = abs(ay-by)
        # hcost = max(dx, dy) * 10

        return int( hcost )


    def draw_all_nodes_for_view(self):
        '''
        Draw each point in OPEN and CLOSE groups on the map, for visulization only
        '''
        # print('open')
        for i in self.nonqueue:
            # print( i, self.open_nodes[i][0], self.open_nodes[i][1], self.open_nodes[i][2])
            self.obstacle_map_view[i[1],i[0]] = [200,100,30]
        # print('close')
        for i in self.queue:
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
                    print('new obstacle at ', x,y)
                    self.obstacle_map[ y, x ] = 0
                    self.obstacle_map_view_backup[ y, x ] = (0,0,0)
                    if (x,y) in self.nonqueue:
                        self.nonqueue.pop((x,y))
                    if (x,y) in self.queue:
                        self.queue.pop((x,y))

                    # next_grid = (self.rbt_next_x, self.rbt_next_y)
                    # x, y = next_grid[0], next_grid[1]
                    # all8rhs = [] 
                    # nb_xys = []  
                    for nb in self.motion: # the 8 direction motions
                        newx = x + nb[0]   # x posi of one of the 8 surrounding grids
                        newy = y + nb[1]   # y posi
                        c_onestep = nb[2]  # cost from (x,y) to this surrounding grid
                        
                        if 0 < newx < self.x_size and 0 < newy < self.y_size and self.obstacle_map[newy,newx] != 0: # if this grid is inside the map and not obstacle 
                            # print('\nfinding rhs: nb of active grid : ',(newx,newy))
                            # g, rhs, h = self.get_grid_content((newx, newy))
                            newgrid = (newx, newy)
                            print('\ncheck new grid ', newgrid)
                            self.update_vertex((newx, newy))

        if len(cleared)!=0:
            # print('remove obstacle from map')
            for i in cleared:
                self.obstacle_map[ i[1], i[0] ] = 255
                self.obstacle_map_view_backup[ i[1], i[0] ] = [255,255,255]






def main():

    # start and goal position
    sx = 16
    sy = 6
    gx = 43 #26 #43  
    gy = 47 #35 #47  
    # gx = 26 #43  
    # gy = 35 #47  

    obstacle_map = cv2.imread( 'map3.png' )
    # obstacle_map = cv2.rotate(obstacle_map, cv2.ROTATE_90_CLOCKWISE)
    
    print('map size: ', obstacle_map.shape)

    astarplanner = lpastar(obstacle_map, sx, sy, gx, gy)



if __name__ == '__main__':
    main()
    


