# Urban Fire Spread Prediction Using Graph Attention Networks

## Setup Instructions

1. Install Requirements
   Python Version Used: 3.12
    ```bash
    pip install -r requirements.txt
    ```

3. Download Required Datasets
    - **Palisades Fire Data**: [https://data.humdata.org/dataset/palisades-fire-building-damage-assessment](#)
    - - Download all 4 files
    - **SCLC Land Cover**: [https://data.mendeley.com/datasets/zykyrtg36g/2](#)
    - **FBFM13 Fuel Model**: [https://landfire.gov/data/FullExtentDownloads](#)
    - - Select CONUS LF 2023 and extract the file LC23_F13_240.tif from the zip file

    Place the datasets inside the `data/` folder.

5. Extract Fire Extent Geometry
    ```bash
    python extract_fire_geojson.py
    ```

6. Generate Vegetation and Wind Layers
    ```bash
    python vegetation_layer.py
    python wind_layers.py
    ```

7. Stack Daily Fire Inputs
    ```bash
    python stack_fire_inputs.py
    ```

8. Train and Validate GNN Model. Note: will take a while to train and produce predictions.
    ```bash
    python gnn.py
    ```
