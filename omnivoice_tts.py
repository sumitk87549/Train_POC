import argparse
import time
import os
import sys

# Validate rich is available for the Iron Man dashboard
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.text import Text
except ImportError:
    print("Error: 'rich' library is required for the dashboard. Please install it using 'pip install rich'")
    sys.exit(1)

console = Console()

def iron_man_header():
    """Prints an Iron Man / J.A.R.V.I.S. styled header."""
    header_text = Text("J.A.R.V.I.S. AUDIO SYNTHESIS PROTOCOL (OMNIVOICE)", style="bold cyan", justify="center")
    console.print(Panel(header_text, border_style="cyan"))

def main():
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. OmniVoice Synthesis Script")
    parser.add_argument("--ref_audio", required=True, help="Path to reference audio file")
    parser.add_argument("--text_file", required=True, help="Path to text file containing Hindi (Devanagari) text")
    parser.add_argument("--out_dir", default="output", help="Output folder to save TTS")
    args = parser.parse_args()

    iron_man_header()
    
    # -------------------------------------------------------------
    # CUSTOMIZE: Paste reference audio's corresponding text here
    # -------------------------------------------------------------
    REF_TEXT = "This is a test reference text for the dummy audio used in cloning."

    console.print(f"[bold cyan]SYSTEM STATUS:[/bold cyan] Initializing core neural modules...")
    
    with Progress(
        SpinnerColumn(spinner_name="aesthetic", style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="[cyan]Loading deep learning dependencies (PyTorch)...", total=None)
        import torch
        import torchaudio
        try:
            from omnivoice import OmniVoice
        except ImportError:
            console.print("[bold red]CRITICAL ERROR: omnivoice package not found. Please install omnivoice.[/bold red]")
            sys.exit(1)
            
    console.print("[bold green]Dependencies successfully integrated.[/bold green]")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        console.print(f"[dim cyan]Output directory calibrated: {args.out_dir}[/dim cyan]")

    with Progress(
        SpinnerColumn(spinner_name="bouncingBar", style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="[cyan]Establishing neural link to k2-fsa/OmniVoice (Allocating CPU tensors)...", total=None)
        model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32)

    console.print("[bold green]OmniVoice Neural Network loaded into central memory.[/bold green]")
    
    try:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            target_text = f.read().strip()
    except Exception as e:
        console.print(f"[bold red]ERROR READING TEXT FILE: {e}[/bold red]")
        sys.exit(1)
        
    console.print(Panel(
        f"[green]{target_text[:150]}...[/green]", 
        title="[cyan]Extracted Text Analysis[/cyan]", 
        border_style="cyan"
    ))

    console.print("[bold cyan]COMMENCING VOCAL SYNTHESIS SEQUENCE...[/bold cyan]")
    start_time = time.time()
    
    with Progress(
        SpinnerColumn(spinner_name="runner", style="bold cyan"),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        progress.add_task(description="[cyan]Synthesizing high-fidelity audio streams...", total=None)
        try:
            audio = model.generate(
                text=target_text,
                ref_audio=args.ref_audio,
                ref_text=REF_TEXT,
            )
        except Exception as e:
            console.print(f"\n[bold red]SYNTHESIS FAILURE ENCOUNTERED: {e}[/bold red]")
            sys.exit(1)
            
    elapsed = time.time() - start_time
    out_filename = os.path.basename(args.text_file).replace(".txt", ".wav")
    if out_filename == os.path.basename(args.text_file):
        out_filename = "output.wav"
    out_path = os.path.join(args.out_dir, out_filename)
    
    # Save the file
    torchaudio.save(out_path, audio[0], 24000)
    
    console.print(Panel(
        f"[bold green]SYNTHESIS COMPLETE[/bold green]\n\n"
        f"Processing Time   :  {elapsed:.2f} seconds\n"
        f"Exported Location :  [yellow]{out_path}[/yellow]",
        title="[cyan]J.A.R.V.I.S. Operation Report[/cyan]",
        border_style="cyan"
    ))

if __name__ == "__main__":
    main()
