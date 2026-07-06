# COLMAP Workflow

This guide uses the dedicated Pixi `colmap` environment and the three task
wrappers:

- `colmap-features`
- `colmap-match`
- `colmap-map`

Each task takes exactly one input: the scene folder.

## 1. Prepare a workspace with images

Create a workspace directory and put your images in an `images/` subfolder:

```bash
mkdir -p /path/to/scene/images
# copy images into /path/to/scene/images
```

Set the folder once and reuse it:

```bash
SCENE_DIR=/path/to/scene
```

## 2. Extract features

Run the feature extraction task:

```bash
pixi run -e colmap colmap-features -- "$SCENE_DIR"
```

## 3. Match sequentially

Run the matching task:

```bash
pixi run -e colmap colmap-match -- "$SCENE_DIR"
```

## 4. Run mapping

Run the mapping task:

```bash
pixi run -e colmap colmap-map -- "$SCENE_DIR"
```

The sparse model will be written under `$SCENE_DIR/sparse/0`.
