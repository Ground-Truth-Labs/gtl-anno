# gtl-anno

assistive annotation tools

## Develop

Use [Google Functions Framework](https://cloud.google.com/functions/docs/functions-framework). Specifically the [Python CLI](https://github.com/GoogleCloudPlatform/functions-framework-python).

``` bash
functions-framework --target propagate_label --debug
```

## Deploy

Using the Google Cloud SDK, [gcloud functions deploy](https://cloud.google.com/sdk/gcloud/reference/functions/deploy)

``` bash
gcloud functions deploy propagate_label --region europe-west2 --allow-unauthenticated --memory 1024MB --trigger-http --runtime python38
```
