"""
Rich progress utilities for long-running operations.

Provides nested progress bars with colors and enhanced formatting
for dataset processing, calibration, and other computationally intensive tasks.
"""

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union
import time

from rich.console import Console
from rich.progress import (
    Progress,
    TaskID,
    BarColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
)
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class RichProgress:
    """Rich progress manager with nested progress bars and custom styling."""
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize progress manager.
        
        Args:
            console: Rich console instance (creates new one if None).
        """
        self.console = console or Console()
        self.progress: Optional[Progress] = None
        self.tasks: Dict[str, TaskID] = {}
        self._active = False
        
    def __enter__(self):
        """Start progress tracking."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop progress tracking."""
        self.stop()
        
    def start(self):
        """Initialize and start progress display."""
        if self._active:
            return
            
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )
        self.progress.start()
        self._active = True
        
    def stop(self):
        """Stop progress display."""
        if not self._active or not self.progress:
            return
            
        self.progress.stop()
        self._active = False
        
    def add_task(
        self,
        name: str,
        description: str,
        total: Optional[int] = None,
        visible: bool = True,
    ) -> str:
        """Add a progress task.
        
        Args:
            name: Unique task identifier.
            description: Human-readable task description.
            total: Total units of work (None for indeterminate).
            visible: Whether task should be visible initially.
            
        Returns:
            Task name for updating progress.
        """
        if not self._active or not self.progress:
            raise RuntimeError("Progress not started")
            
        task_id = self.progress.add_task(
            description=description,
            total=total,
            visible=visible,
        )
        self.tasks[name] = task_id
        return name
        
    def update_task(
        self,
        name: str,
        advance: Optional[int] = None,
        completed: Optional[int] = None,
        description: Optional[str] = None,
        total: Optional[int] = None,
        visible: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Update a progress task.
        
        Args:
            name: Task identifier.
            advance: Units to advance progress by.
            completed: Set absolute completion amount.
            description: Update task description.
            total: Update total work units.
            visible: Update visibility.
            **kwargs: Additional update parameters.
        """
        if not self._active or not self.progress or name not in self.tasks:
            return
            
        task_id = self.tasks[name]
        self.progress.update(
            task_id,
            advance=advance,
            completed=completed,
            description=description,
            total=total,
            visible=visible,
            **kwargs,
        )
        
    def complete_task(self, name: str):
        """Mark a task as completed."""
        if name in self.tasks and self._active and self.progress:
            task_id = self.tasks[name]
            task = self.progress.tasks[task_id]
            if task.total is not None:
                self.progress.update(task_id, completed=task.total)
                
    def remove_task(self, name: str):
        """Remove a task from tracking."""
        if name in self.tasks and self._active and self.progress:
            task_id = self.tasks[name]
            self.progress.remove_task(task_id)
            del self.tasks[name]


@contextmanager
def create_progress(console: Optional[Console] = None):
    """Context manager for creating progress tracking.
    
    Args:
        console: Rich console instance.
        
    Yields:
        RichProgress instance.
    """
    progress = RichProgress(console)
    try:
        progress.start()
        yield progress
    finally:
        progress.stop()


class ProcessingProgress:
    """Specialized progress tracker for data processing operations."""
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize processing progress tracker.
        
        Args:
            console: Rich console instance.
        """
        self.console = console or Console()
        self.progress_manager: Optional[RichProgress] = None
        
    @contextmanager
    def track_dataset_creation(self, datasets: List[str]):
        """Track dataset creation progress with stable display.
        
        Args:
            datasets: List of dataset names to create.
            
        Yields:
            Tuple of (update_dataset function, progress manager for nested tasks).
        """
        with create_progress(self.console) as progress:
            # Main dataset creation progress
            main_task = progress.add_task(
                "main_progress",
                "Dataset creation",
                total=len(datasets),
            )
            
            # Individual step tasks (hidden initially)
            step_tasks = {}
            for i, dataset in enumerate(datasets):
                task_id = progress.add_task(
                    f"step_{i}",
                    f"[dim]{dataset}[/dim]",
                    total=1,
                    visible=False,
                )
                step_tasks[dataset] = task_id
            
            current_step_index = 0
            
            def update_dataset(dataset_name: str, status: str = "processing"):
                """Update progress for a specific dataset."""
                nonlocal current_step_index
                
                if dataset_name in step_tasks:
                    task_id = step_tasks[dataset_name]
                    
                    if status == "processing":
                        # Show current step as active
                        progress.update_task(
                            task_id,
                            description=f"[yellow]●[/yellow] {dataset_name}",
                            visible=True,
                        )
                    elif status == "completed":
                        # Mark as completed and advance main progress
                        progress.update_task(
                            task_id,
                            description=f"[green]✓[/green] {dataset_name}",
                            completed=1,
                        )
                        progress.update_task(main_task, advance=1)
                        current_step_index += 1
                        
                        # Update main task description with current status
                        completed_count = current_step_index
                        total_count = len(datasets)
                        progress.update_task(
                            main_task,
                            description=f"Dataset creation ([green]{completed_count}[/green]/{total_count} complete)",
                        )
                
            # Return both the update function and the progress manager for nesting
            yield update_dataset, progress
            
    @contextmanager
    def track_calibration(self, iterations: int, nested_progress=None):
        """Track calibration progress.
        
        Args:
            iterations: Number of calibration iterations.
            nested_progress: Existing progress manager to add calibration to.
            
        Yields:
            Function to update calibration progress.
        """
        if nested_progress:
            # Add calibration as a nested task in existing progress
            calibration_task = nested_progress.add_task(
                "calibration_nested",
                "Calibration",
                total=iterations,
            )
            
            def update_calibration(
                iteration: int,
                loss_value: Optional[float] = None,
                calculating_loss: bool = False,
            ):
                """Update calibration progress."""
                if calculating_loss:
                    nested_progress.update_task(
                        calibration_task,
                        description=f"[yellow]●[/yellow] Calibration epoch {iteration}/{iterations} • calculating loss",
                    )
                else:
                    loss_text = f" • loss: {loss_value:.6f}" if loss_value else ""
                    nested_progress.update_task(
                        calibration_task,
                        description=f"[blue]●[/blue] Calibration epoch {iteration}/{iterations}{loss_text}",
                        advance=1,
                    )
                    
            yield update_calibration
            
        else:
            # Use standalone progress display
            with create_progress(self.console) as progress:
                main_task = progress.add_task(
                    "calibration",
                    "Running calibration",
                    total=iterations,
                )
                
                def update_calibration(
                    iteration: int,
                    loss_value: Optional[float] = None,
                    calculating_loss: bool = False,
                ):
                    """Update calibration progress."""
                    if calculating_loss:
                        progress.update_task(
                            main_task,
                            description=f"Calibration iteration {iteration}/{iterations} • [yellow]calculating loss[/yellow]",
                        )
                    else:
                        loss_text = f" • loss: {loss_value:.6f}" if loss_value else ""
                        progress.update_task(
                            main_task,
                            description=f"Calibration iteration {iteration}/{iterations}{loss_text}",
                            advance=1,
                        )
                        
                yield update_calibration
            
    @contextmanager
    def track_file_processing(self, files: List[str], operation: str = "processing"):
        """Track file processing operations.
        
        Args:
            files: List of files to process.
            operation: Description of operation being performed.
            
        Yields:
            Function to update file processing progress.
        """
        with create_progress(self.console) as progress:
            main_task = progress.add_task(
                "file_processing",
                f"{operation.title()} {len(files)} files",
                total=len(files),
            )
            
            def update_file(
                filename: str,
                status: str = "processing",
                details: Optional[str] = None,
            ):
                """Update progress for a specific file."""
                details_text = f" • {details}" if details else ""
                progress.update_task(
                    main_task,
                    description=f"{operation.title()} files • [blue]{filename}[/blue] ({status}){details_text}",
                    advance=1 if status == "completed" else 0,
                )
                
            yield update_file


def display_summary_table(
    title: str,
    data: List[Dict[str, Union[str, int, float]]],
    console: Optional[Console] = None,
):
    """Display a formatted summary table.
    
    Args:
        title: Table title.
        data: List of dictionaries with table data.
        console: Rich console instance.
    """
    if not data:
        return
        
    console = console or Console()
    
    table = Table(title=title, show_header=True, header_style="bold magenta")
    
    if data:
        for key in data[0].keys():
            table.add_column(key.replace("_", " ").title())
            
        for row in data:
            table.add_row(*[str(value) for value in row.values()])
            
    console.print(table)


def display_error_panel(
    error_message: str,
    suggestions: Optional[List[str]] = None,
    console: Optional[Console] = None,
):
    """Display an error panel with suggestions.
    
    Args:
        error_message: Main error message.
        suggestions: List of suggested solutions.
        console: Rich console instance.
    """
    console = console or Console()
    
    content = Text(error_message, style="red")
    
    if suggestions:
        content.append("\n\nSuggestions:\n", style="yellow")
        for i, suggestion in enumerate(suggestions, 1):
            content.append(f"  {i}. {suggestion}\n", style="white")
            
    panel = Panel(
        content,
        title="[red]Error[/red]",
        border_style="red",
        expand=False,
    )
    
    console.print(panel)


def display_success_panel(
    message: str,
    details: Optional[Dict[str, Any]] = None,
    console: Optional[Console] = None,
):
    """Display a success panel with optional details.
    
    Args:
        message: Success message.
        details: Dictionary of additional details to display.
        console: Rich console instance.
    """
    console = console or Console()
    
    content = Text(message, style="green")
    
    if details:
        content.append("\n\nDetails:\n", style="blue")
        for key, value in details.items():
            formatted_key = key.replace("_", " ").title()
            content.append(f"  {formatted_key}: {value}\n", style="white")
            
    panel = Panel(
        content,
        title="[green]Success[/green]",
        border_style="green",
        expand=False,
    )
    
    console.print(panel)