video_id=$(
  curl -sS -X POST http://127.0.0.1:30010/v1/videos \
    -H "Content-Type: application/json" \
    -d '{
      "model": "/model",
      "prompt": "A red fox trots through a snowy forest. Clear footsteps crunch in the snow, with cold wind blowing through the pine trees.",
      "seconds": 5,
      "task": "t2va",
      "conditions": [],
      "target": {
        "short_edge": 768,
        "aspect_ratio": "16:9",
        "duration_seconds": 5.0
      },
      "num_outputs_per_prompt": 1,
      "num_inference_steps": 50,
      "flow_shift": 12.0,
      "audio_flow_shift": 3.0,
      "quality": "lossless",
      "seed": 42
    }' |
    jq -r '.id'
)

echo "video_id=${video_id}"

while true; do
  response=$(curl -sS "http://127.0.0.1:30010/v1/videos/${video_id}")
  status=$(echo "$response" | jq -r '.status')

  echo "$(date '+%H:%M:%S') status=${status}"

  if [ "$status" = "completed" ]; then
    echo "$response" | jq
    break
  fi

  if [ "$status" = "failed" ]; then
    echo "$response" | jq
    exit 1
  fi

  sleep 2
done

curl -sS -L \
  "http://127.0.0.1:30010/v1/videos/${video_id}/content" \
  -o fox.mp4

ls -lh fox.mp4
