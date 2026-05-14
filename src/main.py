import yaml


def load_config():
    with open("config/params.yaml", "r") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()

    print("Configuración cargada correctamente")
    print(config)


if __name__ == "__main__":
    main()