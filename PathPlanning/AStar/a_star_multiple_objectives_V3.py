import math
import matplotlib.pyplot as plt

class AStarPlanner:
    def __init__(self, ox, oy, resolution, rr, w1=1.0, w2=1.0, w3=1.0):
        """
        A* grid based planning
        :param ox: x position list of Obstacles [m]
        :param oy: y position list of Obstacles [m]
        :param resolution: grid resolution [m]
        :param rr: robot radius [m]
        :param w1: weight for path length
        :param w2: weight for clearance
        :param w3: weight for energy cost
        """
        self.resolution = resolution
        self.rr = rr
        self.ox = ox
        self.oy = oy
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0  # Gitterbreite und -höhe
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index

    def planning(self, sx, sy, gx, gy):
        """
        A star path search
        :param sx: start x position [m]
        :param sy: start y position [m]
        :param gx: goal x position [m]
        :param gy: goal y position [m]
        :return: rx, ry - path
        """
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if not open_set:
                print("Open set is empty")
                break

            c_id = min(open_set, key=lambda o: open_set[o].cost +
                       self.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

            # Animation der aktuellen Suche
            if True:
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                # Pause für die Animation
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            del open_set[c_id]
            closed_set[c_id] = current

            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                node.cost += (self.w2 / self.calc_clearance(node) +
                              self.w3 * self.calc_energy_cost(current, node))

                if n_id not in open_set or open_set[n_id].cost > node.cost:
                    open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)
        return rx, ry

    def calc_clearance(self, node):
        """ Calculate the clearance from obstacles for a node. """
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)
        min_dist = float("inf")
        for ox, oy in zip(self.ox, self.oy):
            d = math.hypot(px - ox, py - oy)
            if d < min_dist:
                min_dist = d
        return min_dist

    def calc_energy_cost(self, current, neighbor):
        """ Calculate the energy cost based on turning angle. """
        dx1, dy1 = current.x - neighbor.x, current.y - neighbor.y
        dx2, dy2 = neighbor.x - current.x, neighbor.y - current.y
        angle = abs(math.atan2(dy2, dx2) - math.atan2(dy1, dx1))
        return angle

    def calc_final_path(self, goal_node, closed_set):
        """ Construct the final path from start to goal. """
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)],[
                  self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index
        return rx[::-1], ry[::-1]
    
    @staticmethod
    def calc_heuristic(n1, n2):
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def calc_grid_position(self, index, min_pos):
        return index * self.resolution + min_pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x or py < self.min_y or px >= self.max_x or py >= self.max_y:
            return False

        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self,ox, oy):
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

        self.obstacle_map = [[False for _ in range(self.y_width)] for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for ox, oy in zip(self.ox, self.oy):
                    d = math.hypot(ox - x, oy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        """
        Definiert Bewegungsmöglichkeiten: Geradeaus, Diagonal mit Kosten.
        """
        # dx, dy, cost
        motion = [[1, 0, 1],  # Rechts
                  [0, 1, 1],  # Hoch
                  [-1, 0, 1],  # Links
                  [0, -1, 1],  # Runter
                  [-1, -1, math.sqrt(2)],  # Diagonal links unten
                  [-1, 1, math.sqrt(2)],  # Diagonal links oben
                  [1, -1, math.sqrt(2)],  # Diagonal rechts unten
                  [1, 1, math.sqrt(2)]]  # Diagonal rechts oben

        return motion


def main():
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    sx, sy = 10.0, 10.0
    gx, gy = 50.0, 50.0
    grid_size = 2.0
    robot_radius = 1.0

    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "og")
    plt.plot(gx, gy, "xb")
    plt.grid(True)
    plt.axis("equal")
    
    planner = AStarPlanner(ox, oy, grid_size, robot_radius, w1=1.0, w2=3.0, w3=0.5)
    rx, ry = planner.planning(sx, sy, gx, gy)

    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "og")
    plt.plot(gx, gy, "xb")
    plt.plot(rx, ry, "-r")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

if __name__ == '__main__':
    main()
