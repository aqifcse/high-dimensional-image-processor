# High-Dimensional Image Processing Microservice

This project is a microservice designed for high-dimensional image processing, providing an API for image manipulation and analysis. It is built using FastAPI and includes various functionalities for handling images, performing analysis, and returning results.

## Setup Instructions

**Clone the repository:**
   ```
   git clone <repository-url>
   cd high-dimensional-image-processor
   ```

### Using Docker Compose
Install Docker Desktop (latest) in the local machine and start it.
To run the application in a Docker container, execute the following commands from the project root:

1. **Build and run the container:**
```
docker-compose up --build
```

2. **Stop the container:**
```
docker-compose down
```
The API will be available at `http://localhost:8000` (or the port specified in your configuration).
## Example API Calls

### Upload an Image

Using `curl`:

```
curl -X POST http://localhost:8000/upload/ -F "file=@path/to/image"
```

### Retrieve Metadata

Using `curl`:

```
curl -X GET http://localhost:8000/metadata/<filename>
```

### Extract Image Slice

Using `curl`:

```
curl -X POST http://localhost:8000/extract-slice/ -H "Content-Type: application/json" -d '{"filename": "<filename>", "slice_index": <index>}'
```

### Analyze Uploaded Image

Using `curl`:

```
curl -X POST http://localhost:8000/analyze/ -H "Content-Type: application/json" -d '{"filename": "<filename>"}'
```

### Get Image Statistics

Using `curl`:

```
curl -X GET http://localhost:8000/statistics/<filename>
```

### Segment Image

Using `curl`:

```
curl -X POST http://localhost:8000/segment/ -H "Content-Type: application/json" -d '{"filename": "<filename>"}'
```

## Testing

The project includes unit tests to ensure core functionalities are working correctly. To run the tests, use:

```
docker compose exec web pytest --cov=src --cov-report term-missing
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
