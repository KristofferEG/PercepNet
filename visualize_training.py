#!/usr/bin/env python3
"""
PercepNet Training Results Visualization
Parses tensorboard logs and creates comprehensive training visualizations
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
import argparse

# Set style for better looking plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_tensorboard_data(logdir):
    """Load all scalar data from tensorboard logs in the given directory"""
    
    print(f"Loading tensorboard data from: {logdir}")
    
    # Find all event files
    event_files = glob.glob(os.path.join(logdir, "events.out.tfevents.*"))
    print(f"Found {len(event_files)} event files")
    
    all_data = {}
    
    for event_file in event_files:
        print(f"Processing: {os.path.basename(event_file)}")
        
        # Load the event file
        ea = EventAccumulator(event_file)
        ea.Reload()
        
        # Get all scalar tags
        scalar_tags = ea.Tags()['scalars']
        print(f"Available scalar tags: {scalar_tags}")
        
        for tag in scalar_tags:
            scalar_events = ea.Scalars(tag)
            
            # Extract data
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            timestamps = [event.wall_time for event in scalar_events]
            
            if tag not in all_data:
                all_data[tag] = {'steps': [], 'values': [], 'timestamps': []}
            
            all_data[tag]['steps'].extend(steps)
            all_data[tag]['values'].extend(values)
            all_data[tag]['timestamps'].extend(timestamps)
    
    # Sort data by steps for each tag
    for tag in all_data:
        sorted_indices = np.argsort(all_data[tag]['steps'])
        all_data[tag]['steps'] = [all_data[tag]['steps'][i] for i in sorted_indices]
        all_data[tag]['values'] = [all_data[tag]['values'][i] for i in sorted_indices]
        all_data[tag]['timestamps'] = [all_data[tag]['timestamps'][i] for i in sorted_indices]
    
    return all_data

def create_training_plots(data, output_dir):
    """Create comprehensive training visualization plots"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine figure layout based on available metrics
    available_metrics = list(data.keys())
    print(f"Creating plots for metrics: {available_metrics}")
    
    # 1. Loss curves
    loss_metrics = [tag for tag in available_metrics if 'loss' in tag.lower()]
    if loss_metrics:
        plt.figure(figsize=(15, 10))
        
        n_loss_plots = len(loss_metrics)
        n_cols = min(3, n_loss_plots)
        n_rows = (n_loss_plots + n_cols - 1) // n_cols
        
        for i, metric in enumerate(loss_metrics):
            plt.subplot(n_rows, n_cols, i + 1)
            steps = data[metric]['steps']
            values = data[metric]['values']
            
            plt.plot(steps, values, linewidth=2, alpha=0.8)
            plt.title(f'{metric}', fontsize=12, fontweight='bold')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss Value')
            plt.grid(True, alpha=0.3)
            
            # Add trend line
            if len(steps) > 10:
                z = np.polyfit(steps, values, 1)
                p = np.poly1d(z)
                plt.plot(steps, p(steps), "--", alpha=0.5, color='red', 
                        label=f'Trend: {z[0]:.2e}x + {z[1]:.2f}')
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'loss_curves.pdf'), bbox_inches='tight')
        plt.show()
    
    # 2. Learning rate schedule
    lr_metrics = [tag for tag in available_metrics if 'lr' in tag.lower() or 'learning_rate' in tag.lower()]
    if lr_metrics:
        plt.figure(figsize=(12, 6))
        
        for metric in lr_metrics:
            steps = data[metric]['steps']
            values = data[metric]['values']
            plt.semilogy(steps, values, linewidth=2, label=metric, alpha=0.8)
        
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Training Steps')
        plt.ylabel('Learning Rate (log scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'learning_rate.pdf'), bbox_inches='tight')
        plt.show()
    
    # 3. Validation/Evaluation metrics
    eval_metrics = [tag for tag in available_metrics 
                   if any(keyword in tag.lower() for keyword in ['val', 'eval', 'dev', 'test', 'accuracy', 'f1'])]
    if eval_metrics:
        plt.figure(figsize=(15, 10))
        
        n_eval_plots = len(eval_metrics)
        n_cols = min(3, n_eval_plots)
        n_rows = (n_eval_plots + n_cols - 1) // n_cols
        
        for i, metric in enumerate(eval_metrics):
            plt.subplot(n_rows, n_cols, i + 1)
            steps = data[metric]['steps']
            values = data[metric]['values']
            
            plt.plot(steps, values, 'o-', linewidth=2, markersize=4, alpha=0.8)
            plt.title(f'{metric}', fontsize=12, fontweight='bold')
            plt.xlabel('Training Steps')
            plt.ylabel('Value')
            plt.grid(True, alpha=0.3)
            
            # Highlight best value
            if values:
                best_idx = np.argmax(values) if 'loss' not in metric.lower() else np.argmin(values)
                best_step = steps[best_idx]
                best_value = values[best_idx]
                plt.axvline(x=best_step, color='red', linestyle='--', alpha=0.5)
                plt.text(best_step, best_value, f'Best: {best_value:.4f}\n@{best_step}', 
                        fontsize=9, ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'evaluation_metrics.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'evaluation_metrics.pdf'), bbox_inches='tight')
        plt.show()
    
    # 4. All metrics overview (if not too many)
    if len(available_metrics) <= 12:
        plt.figure(figsize=(20, 15))
        
        n_plots = len(available_metrics)
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        for i, metric in enumerate(available_metrics):
            plt.subplot(n_rows, n_cols, i + 1)
            steps = data[metric]['steps']
            values = data[metric]['values']
            
            plt.plot(steps, values, linewidth=2, alpha=0.8)
            plt.title(f'{metric}', fontsize=10, fontweight='bold')
            plt.xlabel('Steps')
            plt.ylabel('Value')
            plt.grid(True, alpha=0.3)
            
            # Add final value annotation
            if values:
                final_value = values[-1]
                final_step = steps[-1]
                plt.text(0.02, 0.98, f'Final: {final_value:.4f}', 
                        transform=plt.gca().transAxes, fontsize=8,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'all_metrics_overview.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'all_metrics_overview.pdf'), bbox_inches='tight')
        plt.show()
    
    # 5. Create summary statistics
    create_summary_report(data, output_dir)

def create_summary_report(data, output_dir):
    """Create a summary report of training metrics"""
    
    report_path = os.path.join(output_dir, 'training_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("PercepNet Training Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        for metric in sorted(data.keys()):
            steps = data[metric]['steps']
            values = data[metric]['values']
            
            if not values:
                continue
                
            f.write(f"Metric: {metric}\n")
            f.write(f"  Total logged points: {len(values)}\n")
            f.write(f"  Training steps: {min(steps)} to {max(steps)}\n")
            f.write(f"  Final value: {values[-1]:.6f}\n")
            f.write(f"  Best value: {max(values):.6f} (higher better) or {min(values):.6f} (lower better)\n")
            f.write(f"  Mean: {np.mean(values):.6f}\n")
            f.write(f"  Std: {np.std(values):.6f}\n")
            
            # Calculate improvement
            if len(values) > 1:
                initial_value = values[0]
                final_value = values[-1]
                improvement = ((final_value - initial_value) / abs(initial_value)) * 100
                f.write(f"  Change from start: {improvement:+.2f}%\n")
            
            f.write("\n")
    
    print(f"Summary report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize PercepNet training results')
    parser.add_argument('--logdir', 
                       default='/ceph/home/student.aau.dk/ar01mf/PercepNet/training_set_sept12_500h/exp_erbfix_x30_snr45_rmax99',
                       help='Path to tensorboard log directory')
    parser.add_argument('--output', 
                       default='/ceph/home/student.aau.dk/ar01mf/PercepNet/training_visualizations',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    print("PercepNet Training Results Visualization")
    print("=" * 50)
    
    # Load data
    data = load_tensorboard_data(args.logdir)
    
    if not data:
        print("No data found in tensorboard logs!")
        return
    
    print(f"\nLoaded data for {len(data)} metrics")
    
    # Create visualizations
    create_training_plots(data, args.output)
    
    print(f"\nVisualizations saved to: {args.output}")
    print("Generated files:")
    for file in glob.glob(os.path.join(args.output, "*")):
        print(f"  - {os.path.basename(file)}")

if __name__ == "__main__":
    main()