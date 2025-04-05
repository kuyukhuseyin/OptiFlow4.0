import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import itertools
import time


class OptiFlow:
    def __init__(self, root):
        self.root = root
        self.root.title("OptiFlow4.0")
        self.root.geometry("1000x700")

        # Variables
        self.algorithm_var = tk.StringVar(value="Johnson")
        self.num_jobs_var = tk.IntVar(value=3)
        self.num_machines_var = tk.IntVar(value=2)
        self.processing_times = []
        self.schedule = []
        self.makespan = 0

        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Algorithm selection
        algo_frame = ttk.LabelFrame(main_frame, text="Algorithm Selection", padding="10")
        algo_frame.pack(fill=tk.X, pady=5)

        ttk.Radiobutton(algo_frame, text="Johnson Algorithm (n×2)",
                        variable=self.algorithm_var, value="Johnson").pack(anchor=tk.W)
        ttk.Radiobutton(algo_frame, text="CDS Algorithm (n×m)",
                        variable=self.algorithm_var, value="CDS").pack(anchor=tk.W)
        ttk.Radiobutton(algo_frame, text="Branch and Bound Algorithm (n×m)",
                        variable=self.algorithm_var, value="BranchBound").pack(anchor=tk.W)

        # Problem size
        size_frame = ttk.LabelFrame(main_frame, text="Problem Size", padding="10")
        size_frame.pack(fill=tk.X, pady=5)

        ttk.Label(size_frame, text="Number of Jobs (n):").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(size_frame, from_=2, to=10, textvariable=self.num_jobs_var).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(size_frame, text="Number of Machines (m):").grid(row=1, column=0, sticky=tk.W)
        machine_spin = ttk.Spinbox(size_frame, from_=2, to=5, textvariable=self.num_machines_var)
        machine_spin.grid(row=1, column=1, sticky=tk.W)

        # Processing times input
        self.times_frame = ttk.LabelFrame(main_frame, text="Processing Times", padding="10")
        self.times_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="Enter Processing Times", command=self.setup_processing_times).pack(side=tk.LEFT,
                                                                                                        padx=5)
        ttk.Button(button_frame, text="Create a Schedule", command=self.run_scheduling).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_all).pack(side=tk.LEFT, padx=5)

        # Results
        self.result_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Initially setup processing times
        self.setup_processing_times()

    def setup_processing_times(self):
        # Clear previous widgets
        for widget in self.times_frame.winfo_children():
            widget.destroy()

        n = self.num_jobs_var.get()
        m = self.num_machines_var.get()

        # Create headers
        ttk.Label(self.times_frame, text="Job No.").grid(row=0, column=0, padx=5, pady=2)
        for j in range(m):
            ttk.Label(self.times_frame, text=f"M{j + 1}").grid(row=0, column=j + 1, padx=5, pady=2)

        # Create entry widgets
        self.time_entries = []
        for i in range(n):
            ttk.Label(self.times_frame, text=f"Job {i + 1}").grid(row=i + 1, column=0, padx=5, pady=2)
            row_entries = []
            for j in range(m):
                entry = ttk.Entry(self.times_frame, width=5)
                entry.grid(row=i + 1, column=j + 1, padx=5, pady=2)
                entry.insert(0, "0")
                row_entries.append(entry)
            self.time_entries.append(row_entries)

    def get_processing_times(self):
        n = self.num_jobs_var.get()
        m = self.num_machines_var.get()
        times = []

        for i in range(n):
            job_times = []
            for j in range(m):
                try:
                    time = int(self.time_entries[i][j].get())
                    if time < 0:
                        raise ValueError
                    job_times.append(time)
                except ValueError:
                    messagebox.showerror("Error", f"Invalid duration for Job {i + 1}, Machine {j + 1}!")
                    return None
            times.append(job_times)

        return times

    def run_scheduling(self):
        algorithm = self.algorithm_var.get()
        n = self.num_jobs_var.get()
        m = self.num_machines_var.get()
        times = self.get_processing_times()

        if not times:
            return

        # Validate algorithm selection
        if algorithm == "Johnson" and m != 2:
            messagebox.showerror("Error", "Johnson Algorithm is only valid for 2 machines!")
            return
        elif algorithm == "CDS" and m < 2:
            messagebox.showerror("Error", "At least 2 machines are required for the CDS Algorithm!")
            return
        elif algorithm == "BranchBound" and (n > 5 or m > 3):
            if not messagebox.askyesno("Error",
                                       "Branch-and-Bound algorithm can be slow for large problems. Do you want to continue??"):
                return

        # Run selected algorithm
        start_time = time.time()

        if algorithm == "Johnson":
            self.schedule, self.makespan = self.johnson_algorithm(times)
        elif algorithm == "CDS":
            self.schedule, self.makespan = self.cds_algorithm(times)
        elif algorithm == "BranchBound":
            self.schedule, self.makespan = self.branch_and_bound(times)

        end_time = time.time()
        computation_time = end_time - start_time

        # Display results
        self.display_results(computation_time)

    def johnson_algorithm(self, processing_times):
        n = len(processing_times)
        jobs = []

        for i in range(n):
            time_m1 = processing_times[i][0]
            time_m2 = processing_times[i][1]
            min_time = min(time_m1, time_m2)

            if time_m1 <= time_m2:
                group = 1  # Schedule early
            else:
                group = 2  # Schedule late

            jobs.append({'id': i, 'm1': time_m1, 'm2': time_m2, 'group': group, 'min_time': min_time})

        # Sort jobs
        group1 = [job for job in jobs if job['group'] == 1]
        group1.sort(key=lambda x: x['m1'])

        group2 = [job for job in jobs if job['group'] == 2]
        group2.sort(key=lambda x: x['m2'], reverse=True)

        schedule = group1 + group2
        schedule_order = [job['id'] for job in schedule]

        # Calculate makespan
        makespan, _ = self.calculate_makespan(schedule_order, processing_times)

        return schedule_order, makespan

    def cds_algorithm(self, processing_times):
        n = len(processing_times)
        m = len(processing_times[0])
        best_order = None
        best_makespan = float('inf')

        for k in range(1, m):
            # Create artificial machines
            artificial_m1 = []
            artificial_m2 = []

            for job in processing_times:
                sum_m1 = sum(job[:k])
                sum_m2 = sum(job[-k:])
                artificial_m1.append(sum_m1)
                artificial_m2.append(sum_m2)

            # Apply Johnson's algorithm
            johnson_input = [[artificial_m1[i], artificial_m2[i]] for i in range(n)]
            current_order, _ = self.johnson_algorithm(johnson_input)

            # Calculate makespan for this order
            makespan, _ = self.calculate_makespan(current_order, processing_times)

            if makespan < best_makespan:
                best_makespan = makespan
                best_order = current_order

        return best_order, best_makespan

    def branch_and_bound(self, processing_times):
        n = len(processing_times)
        m = len(processing_times[0])
        best_order = None
        best_makespan = float('inf')

        # Generate all possible permutations
        for order in itertools.permutations(range(n)):
            makespan, _ = self.calculate_makespan(order, processing_times)

            if makespan < best_makespan:
                best_makespan = makespan
                best_order = order

        return list(best_order), best_makespan

    def calculate_makespan(self, order, processing_times):
        n = len(order)
        m = len(processing_times[0])

        # Initialize start and end times
        start_times = [[0 for _ in range(m)] for _ in range(n)]
        end_times = [[0 for _ in range(m)] for _ in range(n)]

        # Calculate for first job
        for j in range(m):
            if j == 0:
                start_times[0][j] = 0
            else:
                start_times[0][j] = end_times[0][j - 1]
            end_times[0][j] = start_times[0][j] + processing_times[order[0]][j]

        # Calculate for remaining jobs
        for i in range(1, n):
            for j in range(m):
                if j == 0:
                    start_times[i][j] = end_times[i - 1][j]
                else:
                    start_times[i][j] = max(end_times[i - 1][j], end_times[i][j - 1])

                end_times[i][j] = start_times[i][j] + processing_times[order[i]][j]

        makespan = end_times[-1][-1]
        schedule_details = []

        for i in range(n):
            job_details = []
            for j in range(m):
                job_details.append({
                    'start': start_times[i][j],
                    'end': end_times[i][j],
                    'machine': j,
                    'job': order[i]
                })
            schedule_details.append(job_details)

        return makespan, schedule_details

    def display_results(self, computation_time):
        # Clear previous results
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        n = self.num_jobs_var.get()
        m = self.num_machines_var.get()
        times = self.get_processing_times()

        if not times:
            return

        # Calculate detailed schedule
        makespan, schedule_details = self.calculate_makespan(self.schedule, times)

        # Display schedule order
        order_text = "Optimal Order: " + ", ".join([f"İş {i + 1}" for i in self.schedule])
        ttk.Label(self.result_frame, text=order_text, font=('Arial', 10, 'bold')).pack(anchor=tk.W)

        # Display makespan and computation time
        info_text = f"Makespan: {makespan} | Calculation Time: {computation_time:.4f} second"
        ttk.Label(self.result_frame, text=info_text).pack(anchor=tk.W)

        # Create Gantt chart
        self.create_gantt_chart(schedule_details, m)

    def create_gantt_chart(self, schedule_details, num_machines):
        fig, ax = plt.subplots(figsize=(10, 4))

        colors = plt.cm.tab20.colors
        machines = [f"Machine {i + 1}" for i in range(num_machines)]
        y_ticks = np.arange(num_machines)

        # Create bars for each job on each machine
        for job_schedule in schedule_details:
            for operation in job_schedule:
                job_num = operation['job'] + 1
                machine = operation['machine']
                start = operation['start']
                duration = operation['end'] - operation['start']

                ax.barh(machine, duration, left=start, height=0.5,
                        color=colors[job_num % len(colors)], edgecolor='black',
                        label=f'İş {job_num}')

                # Add text label
                ax.text(start + duration / 2, machine, f'J{job_num}',
                        ha='center', va='center', color='white', fontweight='bold')

        # Customize the chart
        ax.set_xlabel('Time')
        ax.set_ylabel('Machines')
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(machines)
        ax.set_title('Gantt Chart - Job Scheduling')
        ax.grid(True, which='both', axis='x', linestyle='--', alpha=0.7)

        # Add legend (without duplicates)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title='Jobs', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Adjust layout
        plt.tight_layout()

        # Embed the chart in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    def clear_all(self):
        self.processing_times = []
        self.schedule = []
        self.makespan = 0

        # Clear results frame
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        # Reset processing times entries
        self.setup_processing_times()


if __name__ == "__main__":
    root = tk.Tk()
    app = OptiFlow(root)
    root.mainloop()