#!/usr/bin/env python3
"""
Intent Distillation Tool CLI
Knowledge distillation for intent classification training data generation
Based on easy-dataset's distillation methodology
"""
import click
import logging
import json
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.tree import Tree
from rich.table import Table

from src.llm.client import LLMClient
from src.distillers.intent_tag_distiller import IntentTagDistiller, IntentNode
from src.distillers.intent_question_distiller import IntentQuestionDistiller
from src.distillers.intent_conversation_distiller import IntentConversationDistiller
from src.exporters.dataset_exporter import DatasetExporter
from src.utils.config_loader import load_config, validate_config

console = Console()


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("distillation.log"),
            logging.StreamHandler()
        ]
    )


@click.group()
@click.option("--config", "-c", default="config.yaml", help="Path to config file")
@click.option("--log-level", default="INFO", help="Logging level")
@click.pass_context
def cli(ctx, config, log_level):
    """Intent Distillation Tool - Generate training data through knowledge distillation"""
    setup_logging(log_level)

    # Load and validate config
    try:
        cfg = load_config(config)
        validate_config(cfg)
        ctx.obj = cfg
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        raise click.Abort()


@cli.command()
@click.option("--parent", "-p", required=True, help="Parent intent name")
@click.option("--count", "-n", default=10, help="Number of sub-intents to generate")
@click.option("--intent-path", help="Full intent path (optional)")
@click.option("--existing", multiple=True, help="Existing sibling intents to avoid")
@click.option("--output", "-o", help="Output file path")
@click.option("--language", "-l", default="en", help="Language (en/zh)")
@click.option("--model", "-m", default="deepseek", help="Model to use")
@click.pass_context
def distill_tags(ctx, parent, count, intent_path, existing, output, language, model):
    """Generate intent taxonomy (sub-intents for a parent intent)"""
    config = ctx.obj

    # Initialize LLM client
    llm_config = config["llm"].get(model)
    if not llm_config:
        console.print(f"[red]Model '{model}' not configured[/red]")
        raise click.Abort()

    llm_client = LLMClient(llm_config)

    # Initialize distiller
    distiller = IntentTagDistiller(llm_client, language)

    console.print(f"\n[cyan]Distilling sub-intents for:[/cyan] {parent}")
    console.print(f"[dim]Generating {count} sub-intents...[/dim]\n")

    # Distill tags
    with console.status("[bold green]Generating intent tags..."):
        try:
            # Create parent node
            parent_node = IntentNode(name=parent)
            if intent_path:
                parent_node.path = intent_path

            # Distill sub-intents
            sub_intents = distiller.distill_tags(
                parent_intent=parent,
                count=count,
                parent_node=parent_node,
                existing_tags=list(existing) if existing else None
            )

            # Display results
            console.print("[green]✓ Generated intent tags:[/green]\n")
            for intent in sub_intents:
                console.print(f"  • {intent.full_name}")

            # Save to file if requested
            if output:
                intent_data = [
                    {
                        "name": intent.name,
                        "number": intent.number,
                        "full_name": intent.full_name,
                        "parent": parent
                    }
                    for intent in sub_intents
                ]

                with open(output, "w") as f:
                    json.dump(intent_data, f, indent=2, ensure_ascii=False)

                console.print(f"\n[green]Saved to {output}[/green]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise click.Abort()


@cli.command()
@click.option("--intent", "-i", required=True, help="Intent name")
@click.option("--intent-path", help="Full intent path")
@click.option("--count", "-n", default=20, help="Number of questions to generate")
@click.option("--existing", "-e", help="File with existing questions (to avoid duplicates)")
@click.option("--output", "-o", help="Output file path")
@click.option("--language", "-l", default="en", help="Language (en/zh)")
@click.option("--model", "-m", default="deepseek", help="Model to use")
@click.pass_context
def distill_questions(ctx, intent, intent_path, count, existing, output, language, model):
    """Generate diverse questions for a specific intent"""
    config = ctx.obj

    # Initialize LLM client
    llm_config = config["llm"].get(model)
    if not llm_config:
        console.print(f"[red]Model '{model}' not configured[/red]")
        raise click.Abort()

    llm_client = LLMClient(llm_config)

    # Load existing questions if provided
    existing_questions = None
    if existing and Path(existing).exists():
        with open(existing, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                existing_questions = [q if isinstance(q, str) else q.get("question", "") for q in data]

    # Initialize distiller
    distiller = IntentQuestionDistiller(llm_client, language)

    console.print(f"\n[cyan]Generating questions for intent:[/cyan] {intent}")
    console.print(f"[dim]Generating {count} diverse questions...[/dim]\n")

    # Create intent node
    intent_node = IntentNode(name=intent)
    if intent_path:
        # Parse path to set hierarchy (simplified)
        intent_node.path = intent_path

    # Distill questions
    with console.status("[bold green]Generating questions..."):
        try:
            questions = distiller.distill_questions(
                intent_node=intent_node,
                count=count,
                existing_questions=existing_questions
            )

            # Display results
            console.print("[green]✓ Generated questions:[/green]\n")
            for i, q in enumerate(questions[:10], 1):  # Show first 10
                console.print(f"  {i}. {q['question']}")

            if len(questions) > 10:
                console.print(f"\n  [dim]... and {len(questions) - 10} more[/dim]")

            # Save to file
            if output:
                with open(output, "w") as f:
                    for q in questions:
                        f.write(json.dumps(q, ensure_ascii=False) + "\n")

                console.print(f"\n[green]Saved {len(questions)} questions to {output}[/green]")
            else:
                console.print(f"\n[yellow]Use --output to save questions[/yellow]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise click.Abort()


@cli.command()
@click.option("--topic", "-t", required=True, help="Root topic/domain")
@click.option("--levels", default=3, help="Taxonomy depth (recommended: 2-3)")
@click.option("--tags-per-level", default=5, help="Tags per level (recommended: 4-8)")
@click.option("--questions-per-tag", default=20, help="Questions per intent (recommended: 20-50)")
@click.option("--leaf-only", is_flag=True, default=True, help="Only generate questions for leaf intents")
@click.option("--output", "-o", required=True, help="Output file path")
@click.option("--language", "-l", default="en", help="Language (en/zh)")
@click.option("--model", "-m", default="deepseek", help="Model to use")
@click.option("--export-taxonomy", help="Export taxonomy tree to file")
@click.pass_context
def distill_auto(ctx, topic, levels, tags_per_level, questions_per_tag, leaf_only, output, language, model, export_taxonomy):
    """Fully automated intent distillation (taxonomy + questions)"""
    config = ctx.obj

    # Initialize LLM client
    llm_config = config["llm"].get(model)
    if not llm_config:
        console.print(f"[red]Model '{model}' not configured[/red]")
        raise click.Abort()

    llm_client = LLMClient(llm_config)

    # Calculate expected counts
    total_tags = sum(tags_per_level ** i for i in range(1, levels + 1))
    leaf_tags = tags_per_level ** levels
    total_questions = (leaf_tags if leaf_only else total_tags) * questions_per_tag

    console.print("\n[bold cyan]Intent Distillation Pipeline[/bold cyan]")
    console.print(f"[dim]{'='*50}[/dim]")
    console.print(f"Topic: [yellow]{topic}[/yellow]")
    console.print(f"Taxonomy: {levels} levels × {tags_per_level} tags/level")
    console.print(f"Expected: ~{total_tags} total intents, ~{leaf_tags} leaf intents")
    console.print(f"Questions: {questions_per_tag} per intent × {leaf_tags if leaf_only else total_tags} intents")
    console.print(f"Total: [green]~{total_questions} training samples[/green]")
    console.print(f"[dim]{'='*50}[/dim]\n")

    try:
        # Stage 1: Build taxonomy
        console.print("[bold]Stage 1/2: Building Intent Taxonomy[/bold]")

        tag_distiller = IntentTagDistiller(llm_client, language)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Building {levels}-level taxonomy...", total=levels)

            root = tag_distiller.build_taxonomy(
                root_topic=topic,
                levels=levels,
                tags_per_level=tags_per_level
            )

            progress.update(task, completed=levels)

        # Display taxonomy tree
        console.print("\n[green]✓ Taxonomy built successfully![/green]\n")

        tree = Tree(f"[bold]{root.name}[/bold]")
        _build_tree_display(tree, root, max_depth=3)  # Show first 3 levels
        console.print(tree)

        # Export taxonomy if requested
        if export_taxonomy:
            taxonomy_data = tag_distiller.export_tree(root)
            with open(export_taxonomy, "w") as f:
                json.dump(taxonomy_data, f, indent=2, ensure_ascii=False)
            console.print(f"\n[dim]Taxonomy saved to {export_taxonomy}[/dim]")

        # Stage 2: Generate questions
        console.print(f"\n[bold]Stage 2/2: Generating Questions[/bold]")

        question_distiller = IntentQuestionDistiller(llm_client, language)

        leaf_intents = tag_distiller.get_leaf_intents(root)
        target_intents = leaf_intents if leaf_only else tag_distiller.export_flat_list(root)

        console.print(f"Generating questions for {len(target_intents)} intents...\n")

        all_questions = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Generating questions...", total=len(target_intents))

            for intent_node in (leaf_intents if leaf_only else tag_distiller._get_all_intents(root)):
                try:
                    questions = question_distiller.distill_questions(
                        intent_node=intent_node,
                        count=questions_per_tag
                    )
                    all_questions.extend(questions)
                    progress.advance(task)
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed for {intent_node.full_name}: {e}[/yellow]")
                    progress.advance(task)

        # Save results
        console.print(f"\n[bold]Saving Results[/bold]")

        with open(output, "w") as f:
            for q in all_questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        # Display summary
        console.print("\n[green]✓ Distillation Complete![/green]\n")

        summary_table = Table(title="Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Count", style="green")

        summary_table.add_row("Total Intents", str(total_tags))
        summary_table.add_row("Leaf Intents", str(len(leaf_intents)))
        summary_table.add_row("Questions Generated", str(len(all_questions)))
        summary_table.add_row("Output File", output)

        console.print(summary_table)

        console.print(f"\n[dim]Use 'python cli.py export -i {output} -o training.json --format alpaca' to export for training[/dim]\n")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--topic", "-t", required=True, help="Root topic/domain")
@click.option("--levels", default=2, help="Taxonomy depth")
@click.option("--tags-per-level", default=5, help="Tags per level")
@click.option("--conversations-per-tag", default=5, help="Conversations per intent")
@click.option("--turns-per-conversation", default=4, help="Turns per conversation (user+assistant pairs)")
@click.option("--transition-rate", default=0.3, type=float, help="Intent transition probability (0-1)")
@click.option("--leaf-only", is_flag=True, default=True, help="Only generate for leaf intents")
@click.option("--output", "-o", required=True, help="Output file path")
@click.option("--language", "-l", default="en", help="Language (en/zh)")
@click.option("--model", "-m", default="deepseek", help="Model to use")
@click.option("--export-taxonomy", help="Export taxonomy tree to file")
@click.option("--scenario", help="Custom conversation scenario description")
@click.pass_context
def distill_conversations(ctx, topic, levels, tags_per_level, conversations_per_tag,
                         turns_per_conversation, transition_rate, leaf_only, output,
                         language, model, export_taxonomy, scenario):
    """Generate multi-turn conversations with intent transitions"""
    config = ctx.obj

    # Initialize LLM client
    llm_config = config["llm"].get(model)
    if not llm_config:
        console.print(f"[red]Model '{model}' not configured[/red]")
        raise click.Abort()

    llm_client = LLMClient(llm_config)

    # Calculate expected counts
    total_tags = sum(tags_per_level ** i for i in range(1, levels + 1))
    leaf_tags = tags_per_level ** levels
    total_conversations = (leaf_tags if leaf_only else total_tags) * conversations_per_tag
    avg_turns_per_conv = turns_per_conversation * 2  # user + assistant

    console.print("\n[bold cyan]Multi-Turn Conversation Distillation[/bold cyan]")
    console.print(f"[dim]{'='*60}[/dim]")
    console.print(f"Topic: [yellow]{topic}[/yellow]")
    console.print(f"Taxonomy: {levels} levels × {tags_per_level} tags/level")
    console.print(f"Expected: ~{total_tags} total intents, ~{leaf_tags} leaf intents")
    console.print(f"Conversations: {conversations_per_tag} per intent")
    console.print(f"Turns: {turns_per_conversation} turns per conversation")
    console.print(f"Intent transitions: {int(transition_rate * 100)}% probability")
    console.print(f"Total: [green]~{total_conversations} conversations[/green]")
    console.print(f"[dim]{'='*60}[/dim]\n")

    try:
        # Stage 1: Build taxonomy
        console.print("[bold]Stage 1/2: Building Intent Taxonomy[/bold]")

        tag_distiller = IntentTagDistiller(llm_client, language)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Building {levels}-level taxonomy...", total=levels)

            root = tag_distiller.build_taxonomy(
                root_topic=topic,
                levels=levels,
                tags_per_level=tags_per_level
            )

            progress.update(task, completed=levels)

        console.print("\n[green]✓ Taxonomy built successfully![/green]\n")

        tree = Tree(f"[bold]{root.name}[/bold]")
        _build_tree_display(tree, root, max_depth=3)
        console.print(tree)

        # Export taxonomy if requested
        if export_taxonomy:
            taxonomy_data = tag_distiller.export_tree(root)
            with open(export_taxonomy, "w") as f:
                json.dump(taxonomy_data, f, indent=2, ensure_ascii=False)
            console.print(f"\n[dim]Taxonomy saved to {export_taxonomy}[/dim]")

        # Stage 2: Generate conversations
        console.print(f"\n[bold]Stage 2/2: Generating Multi-Turn Conversations[/bold]")

        conversation_distiller = IntentConversationDistiller(llm_client, language)

        all_conversations = conversation_distiller.distill_conversations_for_tree(
            root_node=root,
            conversations_per_intent=conversations_per_tag,
            turns_per_conversation=turns_per_conversation,
            transition_rate=transition_rate,
            leaf_only=leaf_only,
            scenario=scenario
        )

        # Save results
        console.print(f"\n[bold]Saving Results[/bold]")

        with open(output, "w") as f:
            for conv in all_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        # Display summary
        console.print("\n[green]✓ Conversation Distillation Complete![/green]\n")

        summary_table = Table(title="Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Count", style="green")

        summary_table.add_row("Total Intents", str(total_tags))
        summary_table.add_row("Leaf Intents", str(leaf_tags))
        summary_table.add_row("Conversations Generated", str(len(all_conversations)))
        summary_table.add_row("Avg Turns per Conversation", str(avg_turns_per_conv))
        summary_table.add_row("Output File", output)

        console.print(summary_table)

        console.print(f"\n[dim]Use 'python cli.py export -i {output} -o training.json --format sharegpt' for ShareGPT format[/dim]\n")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--input", "-i", required=True, help="Input classification results file")
@click.option("--output", "-o", required=True, help="Output dataset file")
@click.option("--format", "-f", default="alpaca", help="Export format (alpaca/sharegpt/json/jsonl/csv)")
@click.option("--split", type=float, help="Train/test split ratio (e.g., 0.8)")
@click.option("--system-prompt", help="System prompt for training data")
@click.option("--mode", help="Export mode for conversations (intent-classification/conversation)")
def export(input, output, format, split, system_prompt, mode):
    """Export distillation results to SLM training format"""

    # Load results
    with open(input, "r") as f:
        if input.endswith(".jsonl"):
            results = [json.loads(line) for line in f]
        else:
            results = json.load(f)

    console.print(f"Loaded {len(results)} samples")

    # Split if requested
    if split:
        split_idx = int(len(results) * split)
        train_results = results[:split_idx]
        test_results = results[split_idx:]

        # Export train and test sets
        train_output = output.replace(".", "_train.")
        test_output = output.replace(".", "_test.")

        DatasetExporter.export(train_results, train_output, format, system_prompt=system_prompt or "", mode=mode)
        DatasetExporter.export(test_results, test_output, format, system_prompt=system_prompt or "", mode=mode)

        console.print(f"[green]Train set ({len(train_results)} samples): {train_output}[/green]")
        console.print(f"[green]Test set ({len(test_results)} samples): {test_output}[/green]")
    else:
        # Export all
        DatasetExporter.export(results, output, format, system_prompt=system_prompt or "", mode=mode)
        console.print(f"[green]Exported {len(results)} samples to {output}[/green]")


def _build_tree_display(tree, node: IntentNode, max_depth: int = 3, current_depth: int = 0):
    """Helper to build Rich tree display"""
    if current_depth >= max_depth:
        if node.children:
            tree.add(f"[dim]... {len(node.children)} more ...[/dim]")
        return

    for child in node.children[:5]:  # Show max 5 children per level
        branch = tree.add(f"{child.full_name}")
        if child.children:
            _build_tree_display(branch, child, max_depth, current_depth + 1)

    if len(node.children) > 5:
        tree.add(f"[dim]... and {len(node.children) - 5} more ...[/dim]")


if __name__ == "__main__":
    cli()
