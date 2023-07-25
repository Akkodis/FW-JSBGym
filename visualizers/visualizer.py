import subprocess


class PlotVisualizer(object):
    def __init__(self, scale: bool) -> None:
        cmd: str = ""
        if scale:
            cmd: str = "python visualizers/attitude_control_telemetry.py --scale"
        else:
            cmd: str = "python visualizers/attitude_control_telemetry.py"
        self.process: subprocess.Popen = subprocess.Popen(cmd, 
                                                          shell=True,
                                                          stdout=subprocess.PIPE,
                                                          stderr=subprocess.STDOUT)
        print("Started PlotVisualizer process with PID: ", self.process.pid)
        while True:
            out = self.process.stdout.readline().decode()
            print(out.strip())
            if "ani = FuncAnimation(plt.gcf(), animate, fargs=(ax, args, ), interval=50, blit=False)" in out:
                print("PlotVisualizer loaded successfully.")
                break


