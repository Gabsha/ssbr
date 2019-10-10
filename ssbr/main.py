import click
import json


@click.group()
def cli():
    pass


@cli.command()
@click.option('-c', '--config', type=click.File(mode='r'))
@click.option('-d', '--dataset')
@click.option('-o', '--output')
def train(config, dataset, output):
    from ssbr.runners.train import train_experiment
    configdata = json.load(config)
    train_experiment(configdata, dataset, output)


if __name__ == '__main__':
    cli()