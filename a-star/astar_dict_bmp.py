
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

        self.obstacle_map_view = obstacle_map.copy()
        # self.obstacle_map_view = 255 - self.obstacle_map_view 
        self.obstacle_map_view[self.sy,self.sx] = [0,255,0]
        self.obstacle_map_view[self.gy,self.gx] = [0,0,255]
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

        self.open_nodes = dict()
        self.close_nodes = dict()

        self.reached_goal = False

        # self.obstacle_map[sy,sx] = 77
        # self.obstacle_map[gy,gx] = 33

        # print(self.obstacle_map)

        self.clear_to_move, self.cannot_moveto = [], [] 
        
        self.get_motion_model()
        
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

                        if self.obstacle_map[newy, newx] != 0 or (newx, newy) in self.close_nodes :
                            # print('aaaaaaa')
                            pass 
                        else:
                            # print('bbbbbbb')
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
                self.obstacle_map_view[y,x] = [200,100,30]
            # print('close')
            for i in self.close_nodes:
                # print(i, self.close_nodes[i][0], self.close_nodes[i][1], self.close_nodes[i][2])
                x = i[0]
                y = i[1]
                self.obstacle_map_view[y,x] = [200,30,150]
            # print(self.obstacle_map_view)

            self.obstacle_map_view[self.sy,self.sx] = [0,255,0]

            self.cvshow_larger(self.obstacle_map_view, self.cvshow_ratio, 50)


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
                self.obstacle_map_view[self.sy, self.sx ] = [0,255,0]

            self.obstacle_map_view[y,x] = [100,250,120]
        
        self.obstacle_map_view[self.sy,self.sx] = [0,255,0]
        self.obstacle_map_view[self.gy,self.gx] = [0,0,255]

        self.cvshow_larger(self.obstacle_map_view, self.cvshow_ratio, 0)

        print(the_path)



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
    
    sx = 22
    sy = 39
    
    gx = 180 #313
    gy = 280 #230

    obstacle_map = cv2.imread( 'map2.bmp' )
    # obstacle_map = cv2.rotate(obstacle_map, cv2.ROTATE_90_CLOCKWISE)

    print('bmp size ', obstacle_map.shape)

    # obstacle_map = np.array(
    #     [
    #         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    #         [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
    #         [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
    #         [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
    #         [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
    #         [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
    #         [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
    #         [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
    #         [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
    #         [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
    #         [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
    #         [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
    #         [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
    #         [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
    #         [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
    #         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    #     ]
    # )

    # cv2.imwrite('map.bmp', obstacle_map*255)


    # obstacle_map = np.array(
    #     [
    #         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    #         [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
    #         [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
    #         [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
    #         [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
    #         [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
    #         [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
    #         [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
    #         [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
    #         [1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1]
    #     ]
    # )    


    astarplanner = astar(obstacle_map, sx, sy, gx, gy)



if __name__ == '__main__':
    main()
    


