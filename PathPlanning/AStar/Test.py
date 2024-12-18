import math
from a_star import AStarPlanner
import matplotlib.pyplot as plt

class MOAStarPlanner(AStarPlanner):
    def __init__(self, ox, oy, resolution, rr, w1=1.0, w2=1.0, w3=1.0):
        """
        Multi-Objective A* Planner

        :param ox: x position list of obstacles
        :param oy: y position list of obstacles
        :param resolution: grid resolution
        :param rr: robot radius
        :param w1: weight for path length
        :param w2: weight for clearance
        :param w3: weight for energy cost
        """
        super().__init__(ox, oy, resolution, rr)
        self.ox = ox  # Speichere die x-Koordinaten der Hindernisse
        self.oy = oy  # Speichere die y-Koordinaten der Hindernisse
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def calc_clearance(self, node):
        """
        Calculate clearance cost as inverse distance to the nearest obstacle.
        """
        min_distance = float('inf')
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)
        for ox, oy in zip(self.ox, self.oy):
            d = math.hypot(ox - px, oy - py)
            if d < min_distance:
                min_distance = d
        return min_distance if min_distance > 0 else 1.0  # Avoid division by zero

    def calc_energy_cost(self, current, neighbor):
        """
        Calculate energy cost based on turn angles.
        """
        if current.parent_index == -1:  # No parent, first node
            return 0.0

        parent = self.closed_set[current.parent_index]
        angle1 = math.atan2(current.y - parent.y, current.x - parent.x)
        angle2 = math.atan2(neighbor.y - current.y, neighbor.x - current.x)
        turn_angle = abs(angle2 - angle1)
        return turn_angle

    def update_node_cost(self, current, neighbor):
        """
        Update the cost of a node based on multi-objective optimization.
        """
        move_cost = self.motion[0][2]  # Assuming uniform move cost
        clearance_cost = 1.0 / self.calc_clearance(neighbor)  # Inverse of clearance
        energy_cost = self.calc_energy_cost(current, neighbor)

        total_cost = (current.cost + move_cost +
                      self.w1 * move_cost +
                      self.w2 * clearance_cost +
                      self.w3 * energy_cost)
        return total_cost

    def planning(self, sx, sy, gx, gy):
        """
        Multi-Objective A* Path Planning
        """
        self.closed_set = dict()
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set = dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty. No solution.")
                break

            # Select node with minimum cost
            c_id = min(open_set, key=lambda o: open_set[o].cost +
                       self.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            del open_set[c_id]
            self.closed_set[c_id] = current

            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 0.0, c_id)
                n_id = self.calc_grid_index(node)
                if not self.verify_node(node):
                    continue

                if n_id in self.closed_set:
                    continue

                new_cost = self.update_node_cost(current, node)

                if n_id not in open_set or open_set[n_id].cost > new_cost:
                    node.cost = new_cost
                    open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, self.closed_set)
        return rx, ry

def main():
    print("Multi-Objective A* Path Planning")

    # Start and goal positions
    sx, sy = 10.0, 10.0
    gx, gy = 50.0, 50.0
    grid_size = 2.0
    robot_radius = 1.0

    # Set obstacle positions
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

    # Run Multi-Objective A*
    mo_a_star = MOAStarPlanner(ox, oy, grid_size, robot_radius, w1=1.0, w2=2.0, w3=0.5)
    rx, ry = mo_a_star.planning(sx, sy, gx, gy)

    # Visualization
    if True:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.plot(rx, ry, "-r")
        plt.grid(True)
        plt.axis("equal")
        plt.show()

if __name__ == '__main__':
    main()
