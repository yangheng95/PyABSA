For resource limitation, we do not provide diversities of checkpoints, we hope you can share your checkpoints with those
who have not enough resource to train their model.

1. Upload your zipped checkpoint to Google Drive **in a shared folder**.
   ![123](../documents/pic/pic1.png)

2. Get the link of your checkpoint.
   ![123](../documents/pic/pic2.png)

3. Register the checkpoint in the [checkpoint_map](../checkpoint_map.json), then make a pull request. We will update the
   checkpoints index as soon as we can, Thanks for your help!

```
"checkpoint name": {
        "id": "your checkpoint link",
        "model": "model name",
        "dataset": "trained dataset",
        "description": "trained equipment",
        "version": "used pyabsa version",
        "author": "name (email)"
      }
```
