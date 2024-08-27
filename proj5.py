import numpy as np
import cv2
import math
from queue import PriorityQueue
import multiprocessing
import time

# time interval for time constraint
interval = 0.1

# robot characteristics
radius = 0.038
l = 0.354

rpm1 = 500
rpm2 = 750

# set up start and goal node here
# angle search determines whether the angle goal needs to be reached or any angle is acceptable as long as the
# x and y values are in range
x_start = 20
y_start = 20
angle_start = 45

x_goal = 400
y_goal = 400
angle_goal = 30
angle_search = False

# set up clearance, radius, step size, and thresholds
# made clearance the radius plus the clearance, this can be altered if the robot wasn't meant to be that far
# from an obstacle
clearance = 10 + l
threshold = 0.5
angle_threshold = 30

# the weight is added to the cost value in new_cost = cost + heuristic
# in a more traditional manner it would be added to the heuristic as well but
# this works the same and I couldn't find an adequate weight for the heuristic or it would be too large
weight = 0.1

# graph dimensions
xdim = 500
ydim = 500

# set goal threshold and make threshold map size for the 0.5 threshold
goal_threshold = 10
x_thresh = int(np.round(xdim/threshold))
y_thresh = int(np.round(ydim/threshold))

# define euclidean function to return distances from two sets of points
def euclidean(coords1, coords2):
    return pow((pow(coords1[0] - coords2[0], 2) + pow(coords1[1] - coords2[1], 2)), 0.5)

# class holds the equations for the obstacles
class Map:
    def __init__(self, clear):
        # rectangles
        self.rect = lambda x, y, t: 350 - clear <= y <= 400 + clear and 0 + t%499 <= x <= 50 + t%499 + clear
        self.rect2 = lambda x, y, t: 200 - clear <= y <= 250 + clear and 100 - clear + t%499 <= x <= 150 + clear + t%499
        self.rect3 = lambda x, y, t: 200 - clear <= y <= 400 + clear and 350 + clear + t%499 <= x <= 400 - clear + t%499

        # added clearance to borders
        self.quad = lambda x, y: x <= 0 + clear or y <= 0 + clear or x >= xdim - clear or y >= ydim - clear

    def is_obstacle(self, coords, t):
        # check if a coordinate is in an obstacle space
        return self.quad(coords[0], coords[1]) or self.rect(coords[0], coords[1], t) or self.rect2(coords[0], coords[1], t) or self.rect3(coords[0], coords[1], t)

# node class to hold coordinates, angle, parent node, and current calculated cost
class Node:
    def __init__(self, value, time, parent=None, steps=None, moveID = 0):
        self.coords = (value[0], value[1])
        self.value = value
        self.t = time
        self.x = value[0]
        self.y = value[1]
        self.angle = value[2]
        self.parent = parent
        self.dist = np.inf
        self.dist_t = np.inf
        self.steps = steps
        self.moveID = moveID

    # altered eq and hash function to allow nodes to be properly implemented as dictionary key and in priority queue
    def __eq__(self, other):
        if self.value == other.value:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return str(self.value)

    def cost(self, UL, UR):
        #rpm to rad/s
        UL = UL*math.pi/30
        UR = UR*math.pi/30
        # Xi, Yi,Thetai: Input point's coordinates
        # Xs, Ys: Start point coordinates for plot function
        # Xn, Yn, Thetan: End point coordintes
        Xi, Yi, Thetai = self.value
        t = 0
        r = radius
        L = l
        dt = 0.1
        Xn = Xi
        Yn = Yi
        Thetan = 3.14 * Thetai / 180
        steps = []
        D = 0
        while t < 1:
            t = t + dt
            Xs = Xn
            Ys = Yn
            steps.append((Xs, Ys))
            Delta_Xn = 0.5 * r * (UL + UR) * math.cos(Thetan) * dt
            Delta_Yn = 0.5 * r * (UL + UR) * math.sin(Thetan) * dt
            Thetan += (r / L) * (UR - UL) * dt
            D = D + math.sqrt(math.pow((0.5 * r * (UL + UR) * math.cos(Thetan) *
                                        dt), 2) + math.pow((0.5 * r * (UL + UR) * math.sin(Thetan) * dt), 2))
            Xn = Xn + Delta_Xn
            Yn = Yn + Delta_Yn
        Thetan = 180 * (Thetan) / 3.14
        return Xn, Yn, Thetan, D, steps

    # move functions based on cost equation and rpm
    def l0_r0(self):
        left = 0
        right = 0
        new_x, new_y, new_angle, cost, steps = self.cost(left, right)
        new_val = (new_x, new_y, new_angle)
        new_t = self.t + 1
        return Node(new_val, new_t, self, steps, 0), cost

    def l0_r1(self):
        left = 0
        right = rpm1
        new_x, new_y, new_angle, cost, steps = self.cost(left, right)
        new_val = (new_x, new_y, new_angle)
        new_t = self.t + 1
        return Node(new_val, new_t, self, steps, 1), cost

    def l0_r2(self):
        left = 0
        right = rpm2
        new_x, new_y, new_angle, cost, steps = self.cost(left, right)
        new_val = (new_x, new_y, new_angle)
        new_t = self.t + 1
        return Node(new_val, new_t, self, steps, 2), cost

    def l1_r0(self):
        left = rpm1
        right = 0
        new_x, new_y, new_angle, cost, steps = self.cost(left, right)
        new_val = (new_x, new_y, new_angle)
        new_t = self.t + 1
        return Node(new_val, new_t, self, steps, 3), cost

    def l1_r1(self):
        left = rpm1
        right = rpm1
        new_x, new_y, new_angle, cost, steps = self.cost(left, right)
        new_val = (new_x, new_y, new_angle)
        new_t = self.t + 1
        return Node(new_val, new_t, self, steps, 4), cost

    def l1_r2(self):
        left = rpm1
        right = rpm2
        new_x, new_y, new_angle, cost, steps = self.cost(left, right)
        new_val = (new_x, new_y, new_angle)
        new_t = self.t + 1
        return Node(new_val, new_t, self, steps, 5), cost

    def l2_r0(self):
        left = rpm2
        right = 0
        new_x, new_y, new_angle, cost, steps = self.cost(left, right)
        new_val = (new_x, new_y, new_angle)
        new_t = self.t + 1
        return Node(new_val, new_t, self, steps, 6), cost

    def l2_r1(self):
        left = rpm2
        right = rpm1
        new_x, new_y, new_angle, cost, steps = self.cost(left, right)
        new_val = (new_x, new_y, new_angle)
        new_t = self.t + 1
        return Node(new_val, new_t, self, steps, 7), cost

    def l2_r2(self):
        left = rpm2
        right = rpm2
        new_x, new_y, new_angle, cost, steps = self.cost(left, right)
        new_val = (new_x, new_y, new_angle)
        new_t = self.t + 1
        return Node(new_val, new_t, self, steps, 8), cost

    # function to generate path from a node all the way to the start
    def gen_path(self):
        traceback = []
        counter = self
        while counter.parent:
            traceback.append(counter)
            counter = counter.parent
        traceback.append(counter)
        traceback.reverse()
        return traceback


# Graph class starts with empty dictionary
# key is a node, value holds list with all the associated edges
# holds obstacle map inside to check
class Graph:
    def __init__(self, map_0):
        self.graph = {}
        self.map = map_0

    # function to use to search for the surroundings for a new node and generate graph
    # adds all move function result nodes to the edges list unless they are part of an obstacle
    def gen_nodes(self, n):
        edges = [(n.l0_r0()), (n.l0_r1()), (n.l0_r2()), (n.l1_r0()), (n.l1_r1()), (n.l1_r2()), (n.l2_r0()), (n.l2_r1()), (n.l2_r2())]
        edges = list(filter(lambda val: not self.map.is_obstacle(val[0].coords, n.t), edges))

        self.graph[n] = edges

# initialize data structures for algorithm



def a_star(start, weight):
    start.parent = None
    q = PriorityQueue()
    object_space = Map(clearance)
    graph = Graph(object_space)
    # make start node and check if start and goal are in valid space, initialize visited list
    if object_space.is_obstacle(start.coords, 0) or object_space.is_obstacle((x_goal, y_goal, angle_goal), 0):
        print("Start or goal in obstacle space")
        return None

    visited = np.zeros((x_thresh, y_thresh, 1 + 360 // angle_threshold))


    # start the priority queue with the start node
    start.dist = 0
    q.put((0, 0, start))

    # counter just kept for the priority queue as it couldn't compare nodes and needed an intermediary value
    j = 1
    # loop while the priority queue has a node and pop it out and get the distance and node
    while not q.empty():
        curr_dist, k, curr_node = q.get()
        # round numbers to find threshold value in the visited matrix
        # if the node matches the value from the visited matrix, ignore it
        # if not seen, add these values to the visited matrix
        time = curr_node.t
        round_x = int(np.round(curr_node.value[0]))
        round_y = int(np.round(curr_node.value[1]))
        if curr_node.value[2] < 0:
            round_angle = 360-((-1*curr_node.value[2]) % 360)
            round_angle = int(round_angle)
        else:
            round_angle = curr_node.value[2] % 360
            round_angle = int(round_angle)

        if visited[round_x, round_y, round_angle//angle_threshold] == 1:
            continue
        visited[round_x, round_y, round_angle//angle_threshold] = 1

        # return node if it is within the distance to goal node
        # angle search boolean declared at top determines if the program needs the given goal angle
        if euclidean(curr_node.value, (x_goal, y_goal)) <= goal_threshold:
            if angle_search:
                if round_angle == angle_goal:

                    return curr_node
            else:
                return curr_node

        # generate edges and nodes around the current node
        graph.gen_nodes(curr_node)

        # loop through all the adjacent nodes and check distance value
        # if lower than current, update the value in the node and add the new distance with the heuristic to the q
        # heuristic determined by euclidean distance from goal node
        # weight also added as explained earlier
        for neighbor, cost in graph.graph[curr_node]:
            heuristic = euclidean(neighbor.value, (x_goal, y_goal))
            new_dist = weight*(curr_dist + cost) + heuristic
            if new_dist < neighbor.dist:
                neighbor.dist = new_dist
                q.put((new_dist, j, neighbor))
                j += 1
    # return Not found if node was not found for some reason
    print("Not found")
    return None


def improve_path(routes):
    # improves route by looping through and changing the weight to value the heuristic less and less
    start, route, weight = routes[-1]
    while True:
        weight += 0.01
        if weight > 1:
            weight = 1
        route = a_star(start, weight).gen_path()
        if not route:
            routes.pop()
        else:
            routes[-1] = (start, route, weight)


def ara_star():
    # start initial path with the set weight here
    start = Node((x_start, y_start, angle_start), 0)
    weight = 0.1
    path = a_star(start, weight)
    route = path.gen_path()
    # collect list of current node and their current path to the goal and keep track of weight
    routes = [(start, route, weight)]

    # loop until goal node is reached
    while len(route) > 1:
        # extract current node and route
        # then find the next node in this route and add this to the ist with the path and weight
        curr, route, weight = routes[-1]
        route = route[1:]
        next = route[0]

        routes.append((next, route, weight))

        # multiprocess the improvement algorithm to improve the route and cancel it after t seconds
        manager = multiprocessing.Manager()
        routes = manager.list(routes)
        improve = multiprocessing.Process(target=improve_path, name="Improve Path", args=(routes, ))
        improve.start()

        time.sleep(interval)
        improve.terminate()
        improve.join()

    return routes

# function called to call the algorithm and visualize it
def visual():
    routes = ara_star()
    # print another statement and return None if invalid algorithm output
    if not routes:
        print("Invalid, select new nodes")
        return None

    # set up array as completely black image for now, will update later
    start_frame = np.zeros((y_thresh + 1, x_thresh + 1, 3), np.uint8)

    # add goal node threshold circle to the image
    cv2.circle(start_frame, (int(x_goal * 2), int(y_thresh - y_goal * 2)), int(goal_threshold * 2), (255, 0, 0), -1)

    # create an obstacle space map with a clearance of 0 to show original obstacle sizes
    # check every block and turn white if it is an obstacle
    clear_visual = 0
    visual_space = Map(clear_visual)
    for x in range(0, x_thresh + 1):
        for y in range(0, y_thresh + 1):
            if visual_space.quad(x * threshold, y * threshold):
                start_frame[y_thresh - y, x] = (0, 0, 255)


    # start a video file to write to
    # these settings worked for me on Ubuntu 20.04 to make a video
    output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 120, (x_thresh+1, y_thresh+1))

    # adds exploration path to video by drawing lines from node to node going through the graph dictionary
    # saves a frame every 40 nodes to save video time
    frame_num = 0
    for entry in routes:
        node, route, weight = entry
        frame = start_frame.copy()
        time = node.t
        x_node, y_node = node.coords
        x_node, y_node = int(np.round(2*x_node)), int(np.round(y_thresh - 2 * y_node))
        cv2.circle(frame, (x_node, y_node), 5, (0, 0, 255), -1)

        # draw moving rectangle
        rect1 = [2*(0 + time%499), 2*(50 + time%499), 700, 800]
        rect2 = [2*(100 + time%499), 2*(150 + time%499), 400, 500]
        rect3 = [2*(350 + time%499), 2*(400 + time%499), 400, 800]

        print(rect1)
        frame[y_thresh-rect1[3]:y_thresh-rect1[2], rect1[0]:rect1[1]] = (255,255,255)
        frame[y_thresh - rect2[3]:y_thresh - rect2[2], rect2[0]:rect2[1]] = (255, 255, 255)
        frame[y_thresh - rect3[3]:y_thresh - rect3[2], rect3[0]:rect3[1]] = (255,255,255)

        i = 0
        prev = None
        for location in route:
            x1, y1 = location.coords
            x1 = int(np.round(2*x1))
            y1 = int(np.round(y_thresh-2*y1))
            if prev:
                x2, y2 = prev.coords
                x2 = int(np.round(2 * x2))
                y2 = int(np.round(y_thresh - 2 * y2))
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            prev = location
        output.write(frame)
    output.release()

visual()