# holodex.py
Barebones async holodex python library with json cache (yes it's bad idc)
```py

from holodex import HolodexClient

client = HolodexClient(API_KEY, jwt="...", )

async def main(self):

    # https://docs.holodex.net/#/paths/~1live/get
    live = await client.fetch_live()

    # https://docs.holodex.net/#/paths/~1videos/get
    videos = await client.fetch_videos()

    # https://docs.holodex.net/#operation/get-v2-channels-channelId
    channel = await client.fetch_channel("UCW5uhrG1eCBYditmhL0Ykjw")

    # https://docs.holodex.net/#operation/get-v2-channels-channelId-clips
    channel_videos = await client.fetch_channel_videos("UCW5uhrG1eCBYditmhL0Ykjw", video_type="videos")
    # OR from model with default video_type
    channel_videos = await channel.fetch_videos()
    channel_clips = await channel.fetch_clips()
    channel_collabs = await channel.fetch_collabs()

    # https://docs.holodex.net/#operation/get-cached-live
    cached = await client.fetch_cached_live("UCW5uhrG1eCBYditmhL0Ykjw", "UCDHABijvPBnJm7F-KlNME3w", "UCvN5h1ShZtc7nly3pezRayg", ...)

    # https://docs.holodex.net/#operation/get-videos-videoId
    video = await client.fetch_video("NdlSHUEVCj8")

    # https://docs.holodex.net/#operation/get-channels
    channels = await client.fetch_channels(...)

    # videoSearch and commentSearch not implemented
```

There's several more functions here that are undocumented endpoints. Any editor functions will require a jwt for a Holodex account with relevant permissions.