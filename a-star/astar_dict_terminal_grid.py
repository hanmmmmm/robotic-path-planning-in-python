
import math
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import time



class astar():
    def __init__(self, obstacle_map, sx, sy, gx, gy):
        

        self.sx = sx
        self.sy = sy
        self.gx = gx
        self.gy = gy
        
        self.obstacle_map = obstacle_map 

        self.x_size = self.obstacle_map.shape[1]
        self.y_size = self.obstacle_map.shape[0]
        

        self.open_nodes = dict()
        self.close_nodes = dict()

        self.reached_goal = False

        # self.obstacle_map[sy,sx] = 77
        # self.obstacle_map[gy,gx] = 33

        # print(self.obstacle_map)

        self.clear_to_move, self.cannot_moveto = [], [] 
        
        self.get_motion_model()
        
        self.obstacle_map_view = self.obstacle_map.copy()

        # self.obstacle_map_view = self.binary2color(self.obstacle_map.copy())
        # self.cvshow_larger(self.obstacle_map_view, 30, 0)

        self.planning()


    def planning(self):
        self.open_nodes = dict()
        self.close_nodes = dict()



        self.open_nodes[(self.sx, self.sy)] = [ 0,0,0, [ self.sx, self.sy ] ]  # G, H, F, parent_grid 
        # self.open_nodes[(self.sx+2, self.sy-5)] = [ 1,1,2, [ self.sx, self.sy ] ]
        # self.open_nodes[(self.sx+1, self.sy-1)] = [ 0,0,0, [ self.sx, self.sy ] ]

        self.reached_goal = False

        while self.reached_goal == False:
            # print('')

            fcosts = []
            gcosts = []
            grids = []
            for i in self.open_nodes:
                # print('i  ',i)
                grids.append(i)
                gcost = self.open_nodes[i][0]
                hcost = self.calc_h_cost([i[0],i[1]],[self.gx, self.gy])
                fcost = gcost + hcost
                fcosts.append( fcost )
                gcosts.append( gcost )
                # print('open-node #', i , 'costs: G  H  F ', gcost, hcost, fcost )
            fcosts_min = min(fcosts)
            fcosts = np.array( fcosts )
            min_fcosts_index = np.where(fcosts == fcosts_min)[0]
            # print('costs: G  H  F ', gcost, hcost, fcost )
            # print(min_fcosts_index)
            # print(grids)

            # for i in range(min_fcosts_index.shape[0]):
            for i in min_fcosts_index:    
                # print('min fcost node', i)
                x = grids[i][0]
                y = grids[i][1]
                gcost = gcosts[i]
                l_node = self.open_nodes.pop( (x,y) )
                self.close_nodes[grids[i]] = l_node

                if x == self.gx and y == self.gy:
                    print('Found it !!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    self.reached_goal = True
                
                for nb in self.motion:
                    # print('### i ',i,'nb ', nb)
                    newx = x + nb[0]
                    newy = y + nb[1]
                    c_onestep = nb[2]

                    if 0 < newx < self.x_size and 0 < newy < self.y_size:

                        if self.obstacle_map[newy, newx] == 1 or (newx, newy) in self.close_nodes :
                            pass 
                        else:
                            if (newx, newy) not in self.open_nodes or (gcost + c_onestep + self.calc_h_cost([ newx, newy ],[self.gx , self.gy])) < self.open_nodes[(newx, newy)][2]:
                                self.open_nodes[(newx, newy)] = [0,0,0,[0,0]]
                                self.open_nodes[(newx,newy)][2] = (gcost + c_onestep) + self.calc_h_cost([ newx, newy ],[self.gx , self.gy])
                                self.open_nodes[(newx,newy)][0] = (gcost + c_onestep)
                                self.open_nodes[(newx,newy)][1] = self.calc_h_cost([ newx, newy ],[self.gx , self.gy])
                                self.open_nodes[(newx,newy)][3] = [x,y] 
                            

            # update map for visul
            # print('open')
            for i in self.open_nodes:
                # print( i, self.open_nodes[i][0], self.open_nodes[i][1], self.open_nodes[i][2])
                x = i[0]
                y = i[1]
                self.obstacle_map_view[y,x] = 7
            # print('close')
            for i in self.close_nodes:
                # print(i, self.close_nodes[i][0], self.close_nodes[i][1], self.close_nodes[i][2])
                x = i[0]
                y = i[1]
                self.obstacle_map_view[y,x] = 8
            # print(self.obstacle_map_view)


            # print( (self.gx, self.gy) in self.close_nodes )

        the_path = []
        the_path.append([self.gx, self.gy])
        
        # print('close')
        # for i in self.close_nodes:
        #     print(i, self.close_nodes[i][0], self.close_nodes[i][1], self.close_nodes[i][2])

        backed = False
        while backed == False:
            x = the_path[-1][0]
            y = the_path[-1][1]
            parent_node = self.close_nodes[ (x,y) ][3]
            the_path.append(parent_node)
            if self.sx == parent_node[0] and self.sy == parent_node[1]:
                backed = True
                # the_path.append([self.sx, self.sy])
                self.obstacle_map[self.sy, self.sx ] = 4

            self.obstacle_map[y,x] = 4
        
        # self.obstacle_map[self.sy,self.sx] = 7
        # self.obstacle_map[self.gy,self.gx] = 2

        print(self.obstacle_map)

        print(the_path)

            # time.sleep(1)

    def calc_h_cost(self, a, b):
        
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

        # print('calc_h_cost ', ax,ay,bx,by, hcost)

        return int( hcost )



    def update_typemap_per_center(self, x,y):
        for i in self.motion:
            newx = x + i[0]
            newy = y + i[1]

            if self.obstacle_map[newy, newx] == 1:
                self.typemap[newy, newx] = 1
            elif self.typemap[newy, newx] == 0:
                self.typemap[newy, newx] = 4


    def update_cost_this_grid(self, x, y):
        parent_costs = []
        cost = []

        if self.gy == y and self.gx == x:
            self.reached_goal = True

        for i in self.motion:
            newx = x + i[0]
            newy = y + i[1]

            

            if self.typemap[newy, newx] == 3:
                cost.append( self.costmap[newy,newx] + i[2] )
                parent_costs.append( self.costmap[newy,newx] )
            # elif self.typemap[newy, newx] != 2:
            #     self.typemap[newy, newx] = 4
        # print('parent costs', parent_costs)
        smallest_parent_cost = min(parent_costs)
        smallest_cost = min(cost)
        
        self.costmap[y,x] = smallest_cost
        # return smallest_parent_cost

    def find_smallest_cost(self, x, y):
        costs = []
        valuable_grids = []
        for i in self.motion:
            newx = x + i[0]
            newy = y + i[1]

            # if self.costmap[newy, newx] != 0:
            costs.append(self.costmap[newy, newx])
            valuable_grids.append( [newx, newy] )
        min_c = min(costs)
        min_c_i = costs.index(min_c)
        return valuable_grids[min_c_i]
        



    def update_one_node(self, x, y, cost):
        clear_to_move_in_update = []
        cannot_moveto_in_update = []
        for i in self.motion:
            newx = x + i[0]
            newy = y + i[1]
            newcost = cost + i[2]

            # print(self.obstacle_map[newy, newx])

            if self.obstacle_map[newy, newx] == 0:
                newcost = newcost
                clear_to_move_in_update.append([newx, newy, newcost])
            elif self.obstacle_map[newy,newx] == 1:
                newcost = -1 
                cannot_moveto_in_update.append([newx, newy, newcost])


        cannot_moveto_in_update.append([x, y, cost])

        return clear_to_move_in_update, cannot_moveto_in_update

                

    def get_motion_model(self):
        # dx, dy, cost
        self.motion = [[1, 0, 10],
                  [0, 1, 10],
                  [-1, 0, 10],
                  [0, -1, 10],
                  [-1, -1, 14],
                  [-1, 1, 14],
                  [1, -1, 14],
                  [1, 1, 14]]

    # def get_motion_model(self):
    #     # dx, dy, cost
    #     self.motion = [[1, 0, 1],
    #               [0, 1, 1],
    #               [-1, 0, 1],
    #               [0, -1, 1],
    #               [-1, -1, math.sqrt(2)],
    #               [-1, 1, math.sqrt(2)],
    #               [1, -1, math.sqrt(2)],
    #               [1, 1, math.sqrt(2)]]

       
    def binary2color(self, img):
        img[img==0] = 255
        img[img==1] = 0
        img[img==2] = 55
        img[img==3] = 150
        
        return img.astype(np.uint8)

    def cvshow_larger(self, img, ratio, t):
        original_y = img.shape[0]
        original_x = img.shape[1]
        enlarged = cv2.resize(img, (original_x*ratio, original_y*ratio),interpolation = cv2.INTER_NEAREST)
        print('img size', img.shape)
        print('large size', enlarged.shape)
        cv2.imshow('plan', enlarged)
        cv2.waitKey(0)








def main():
    # start and goal position
    
    sx = 2
    sy = 7
    
    gx = 13
    gy = 6


    obstacle_map = np.array(
        [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        ]
    )

    cv2.imwrite('map.bmp', obstacle_map*255)


    obstacle_map = np.array(
        [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1]
        ]
    )    

    # plt.plot(sx, sy, "og")
    
    # plt.plot(gx, gy, "xb")

    # plt.grid(True)
    # plt.axis("equal")

    astarplanner = astar(obstacle_map, sx, sy, gx, gy)

    # plt.show()


if __name__ == '__main__':
    main()
    


