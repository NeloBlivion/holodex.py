from operator import attrgetter
import aiohttp, asyncio, re
from datetime import datetime, timezone, timedelta
from json import loads

__all__ = (
    "HolodexClient",
    "PlaceholderCredits",
    "Comment",
    "Stream",
    "Clip",
    "Placeholder",
    "Vtuber",
    "Clipper",
    "Org",
    "Topic"
)

BASE_URL = "https://holodex.net"
API_URL = "/api/v2"

# py-cord utils

def pt(timestamp):
    if timestamp:
        return datetime.fromisoformat(timestamp)
    return None

def fdt(dt, /, style = None):
    if isinstance(dt, datetime.time):
        dt = datetime.datetime.combine(datetime.now(), dt)
    if style is None:
        return f"<t:{int(dt.timestamp())}>"
    return f"<t:{int(dt.timestamp())}:{style}>"

def find(predicate, seq):
    for element in seq:
        if predicate(element):
            return element
    return None

def get(iterable, **attrs):
    _all = all
    attrget = attrgetter

    if len(attrs) == 1:
        k, v = attrs.popitem()
        pred = attrget(k.replace("__", "."))
        for elem in iterable:
            if pred(elem) == v:
                return elem
        return None

    converted = [
        (attrget(attr.replace("__", ".")), value) for attr, value in attrs.items()
    ]

    for elem in iterable:
        if _all(pred(elem) == value for pred, value in converted):
            return elem
    return None

def ed(string):
    return string

# etc

def filter_params(params: dict):
    if not params:
        return None
    return {k: v for k, v in params.items() if v is not None}

def fix_url(url):
    if not url.startswith("/"):
        url = "/" + url
    return API_URL + url

def parse_int(string: str | None) -> int | None:
    if not string:
        return None
    return int(str(string).replace(",", ""))

def parse_time(string: str | None) -> datetime | None:
    if not string:
        return None
    return pt(string.split("+")[0].split(".")[0]).replace(tzinfo=timezone.utc)

async def parse_type(data):
    c = str(data.content_type)
    if "json" in c:
        return await data.json()
    if "text" in c:
        return await data.text()
    return data

class PlaceholderCredits:

    @staticmethod
    def from_data(data):
        if not data: return
        for k, v in data.items():
            name = v.get("name")
            link = v.get("link")
            user = v.get("user")
            return PlaceholderCredits(k, name=name, link=link, user=user)

    def __init__(self, type: str, name: str=None, link: str=None, user: str=None):
        self.type = type
        self.name = name
        self.link = link
        self.user = user

    def __str__(self):
        base = f"<PlaceholderCredits type={self.type} "
        ext = " ".join([f"{attr}={getattr(self, attr)}" for attr in ["name", "link", "user"] if getattr(self, attr)])
        return base + ext + ">"

    def to_dict(self):
        ret = {f"{self.type}": {}}
        if self.name:
            ret[self.type]["name"] = self.name
        if self.link:
            ret[self.type]["link"] = self.link
        if self.user:
            ret[self.type]["user"] = self.user
        return ret
        

class Comment:

    def __init__(self, data: dict):
        self.comment_key: str = data.get("comment_key")
        self.video_id: str = data.get("video_id")
        self.message: str = data.get("message")

class HolodexClient:

    def __init__(self, api_key: str = None, jwt: str = None, cls=None, cache: dict = None):
        self.api_key = api_key
        self.jwt = jwt
        self.cls = cls or Vtuber
        self.headers = None
        self.session = self.generate_session()

        self._vtubers = {}
        self._clippers = {}
        self._orgs = {}
        self._topics = {}
        self._videos: dict[str, BaseVideo] = {}
        self.mapping = {
            "vtuber": self._vtubers,
            "subber": self._clippers,
        }
        self.load_channels(cache or {})

    def get_headers(self):
        headers = {}
        if self.api_key:
            headers["X-APIKEY"] = self.api_key
        if self.jwt:
            headers["Authorization"] = f"BEARER {self.jwt}"
        return headers

    def generate_session(self):
        self.headers = self.get_headers()
        return aiohttp.ClientSession(headers=self.headers)
    
    async def request(self, method, endpoint, base=True, **kwargs):
        body = kwargs.pop("body", None)
        params = kwargs.pop("params", None)
        url = kwargs.pop("url", None) or fix_url(endpoint)
        if base:
            url = BASE_URL + url
        async with self.session.request(method, url, json=body, params=filter_params(params)) as r:
            await r.read()
        return r

    async def static(self, url, base=True, raw=False):
        if base:
            url = BASE_URL + url
        async with self.session.get(url) as r:
            await r.read()
        return await parse_type(r) if raw else r

    async def get(self, endpoint, body: dict = None, params: dict = None, raw=False):
        ret = await self.request("GET", endpoint, body=body, params=params)
        return await parse_type(ret) if raw else ret

    async def post(self, endpoint, body: dict = None, params: dict = None, raw=False):
        ret = await self.request("POST", endpoint, body=body, params=params)
        return await parse_type(ret) if raw else ret

    async def delete(self, endpoint, body: dict = None, params: dict = None, raw=False):
        ret = await self.request("DELETE", endpoint, body=body, params=params)
        return await parse_type(ret) if raw else ret

    @property
    def vtubers(self):
        return list(self._vtubers.values())

    @property
    def clippers(self):
        return list(self._clippers.values())

    @property
    def channels(self):
        return self.vtubers + self.clippers

    @property
    def orgs(self):
        return list(self._orgs.values())

    @property
    def topics(self):
        return list(self._topics.values())

    @property
    def videos(self):
        return list(self._videos.values())

    def load_channels(self, channels: dict):
        for channel_id, channel in channels.items():
            channel["id"] = channel_id
            ch = self.raw_channel(channel)
            self.write_channel(ch)

    def raw_channel(self, data, force_type=None):
        if force_type:
            data["type"] = force_type
        if data.get("type") == "vtuber":
            return self.cls(self, data)
        elif data.get("type") == "subber":
            return Clipper(self, data)

    def write_channel(self, channel):
        self.mapping[channel.type][channel.id] = channel
        if channel.type == "vtuber":
            self.insert_org(channel.org)

    def generate_channel(self, data):
        if not data:
            return None
        if not data.get("type"):
            data["type"] = "vtuber"
        return self.get_channel(data["id"]) or self.raw_channel(data)

    def get_channel(self, id: str):
        return get(self.channels, id=id)

    def find_channel(self, exact=False, **kwargs):
        j = self.find_channels(limit=1, exact=exact, **kwargs)
        if not j:
            return None
        return j[0]

    def lookup(self, name: str):
        return self.find_channel(name=name)

    def get_org(self, name: str):
        return get(self.orgs, name=name)

    def insert_org(self, org):
        if not org:
            return
        if not self.get_org(str(org)) and isinstance(org, Org):
            self._orgs[str(org)] = org

    def get_topic(self, name: str):
        if not name:
            return None
        return find(lambda t: t.id.lower() == name.lower(), self.topics)

    def gen_topic(self, name: str):
        if not name:
            return None
        return find(lambda t: t.id.lower() == name.lower(), self.topics) or Topic.from_str(self, name)

    def insert_topic(self, topic):
        if not topic:
            return
        if not self.get_topic(topic.id) and isinstance(topic, Topic):
            self._topics[topic.id] = topic

    def get_video(self, id: str):
        return get(self.videos, id=id)

    def insert_video(self, video):
        if not video:
            return
        if not self.get_video(str(video)) and isinstance(video, BaseVideo):
            self._videos[video.id] = video

    def find_channels(self, limit=None, exact=False, **kwargs):
        results = []
        l = 0
        pool = kwargs.pop("pool", self.channels)
        if not kwargs:
            return []
        for channel in pool:
            if limit and len(results) >= limit:
                break
            i = 0
            for k, v in kwargs.items():
                if (z := getattr(channel, k, None)) == v:
                    i += 1
                elif not exact and isinstance(v, str) and v.lower() in str(z).lower():
                    i += 1
            if i and i == len(kwargs):
                results.append(channel)
            # if (
            #     (channel.id == id if id else True)
            #     and (name.lower() in channel.name.lower() if name else True)
            #     and (org.lower() in channel.org.name.lower() if org else True)
            # ):
            #     results.append(channel)
        return results

    async def fetch_channel(self, channel_id: str):
        data = await self.get(f"channels/{channel_id}")
        if data.status == 404:
            raise Exception(f"{await data.text()}")
        elif data.status == 200:
            data = await data.json()
            if data.get("type") == "vtuber":
                return self.cls(self, data)
            elif data.get("type") == "subber":
                return Clipper(self, data)
        return data

    async def fetch_channels(
            self,
            limit: int = 25,
            offset: int = 0,
            type: ["vtuber", "subber"] = None,
            lang: str = None,
            order: ["asc", "desc"] = "asc",
            org: str = None,
            sort: str = None,
    ):
        params = {
            "limit": limit,
            "offset": offset,
            "type": type,
            "lang": lang,
            "order": order,
            "org": org,
            "sort": sort,
        }
        data = await self.get(f"channels", params=params)
        if data.status == 404:
            raise Exception(f"{await data.text()}")
        elif data.status == 200:
            channels = []
            data = await data.json()
            for channel in data:
                if channel.get("type") == "vtuber":
                    channels.append(self.cls(self, channel))
                elif channel.get("type") == "subber":
                    channels.append(Clipper(self, channel))
            return channels
        return data

    async def fetch_video(self, video_id: str):
        data = await self.get(f"videos/{video_id}")
        if data.status == 404:
            raise Exception(f"{await data.text()}")
        elif data.status == 200:
            data = await data.json()
            return BaseVideo.from_data(self, data)
        return data

    async def fetch_live(
            self,
            channel_id: str = None,
            id: str = None,
            include: list["clips", "refers", "sources", "simulcasts", "mentions", "description", "live_info", "channel_stats", "songs"] = None,
            limit: int = 25,
            max_upcoming_hours: int = None,
            offset: int = 0,
            mentioned_channel_id: str = None,
            type: ["stream", "clip", "placeholder"] = None,
            order: ["asc", "desc"] = "desc",
            org: str = None,
            sort: str = None,
            status: ["new", "upcoming", "live", "past", "missing"] = None,
            topic: str = None,
            paginated: bool = False,
    ):
        params = {
            "channel_id": channel_id,
            "id": id,
            "include": ",".join(include) if include else None,
            "limit": limit,
            "max_upcoming_hours": max_upcoming_hours,
            "mentioned_channel_id": mentioned_channel_id,
            "status": status,
            "topic": topic,
            "offset": offset,
            "type": type,
            "order": order,
            "org": org,
            "sort": sort,
            "paginated": paginated or None
        }
        data = await self.get(f"live", params=params)
        if data.status == 404:
            raise Exception(f"{await data.text()}")
        elif data.status == 200:
            videos = []
            data = await data.json()
            for video in data:
                videos.append(BaseVideo.from_data(self, video))
            return videos
        return data

    async def fetch_cached_live(self, *channel_ids):
        params = {"channels": ",".join(channel_ids)}
        data = await self.get(f"users/live", params=params)
        if data.status == 404:
            raise Exception(f"{await data.text()}")
        elif data.status == 200:
            videos = []
            data = await data.json()
            for video in data:
                videos.append(BaseVideo.from_data(self, video))
            return videos
        return data

    async def fetch_videos(
            self,
            channel_id: str = None,
            id: str = None,
            include: list["clips", "refers", "sources", "simulcasts", "mentions", "description", "live_info", "channel_stats", "songs"] = None,
            lang: str = None,
            limit: int = 25,
            max_upcoming_hours: int = None,
            offset: int = 0,
            mentioned_channel_id: str = None,
            type: ["stream", "clip", "placeholder"] = None,
            order: ["asc", "desc"] = "desc",
            org: str = None,
            sort: str = None,
            status: ["new", "upcoming", "live", "past", "missing"] = None,
            topic: str = None,
            after: datetime = None,
            before: datetime = None,
            paginated: bool = False,
    ):
        params = {
            "channel_id": channel_id,
            "id": id,
            "include": ",".join(include) if include else None,
            "limit": limit,
            "max_upcoming_hours": max_upcoming_hours,
            "mentioned_channel_id": mentioned_channel_id,
            "status": status,
            "topic": topic,
            "from": after.isoformat() if after else None,
            "to": before.isoformat() if before else None,
            "offset": offset,
            "type": type,
            "lang": lang,
            "order": order,
            "org": org,
            "sort": sort,
            "paginated": paginated or None
        }
        data = await self.get(f"videos", params=params)
        if data.status == 404:
            raise Exception(f"{await data.text()}")
        elif data.status == 200:
            videos = []
            data = await data.json()
            for video in data:
                videos.append(BaseVideo.from_data(self, video))
            return videos
        return data

    async def fetch_channel_videos(
        self,
        channel_id: str,
        video_type: ["clips", "videos", "collabs"],
        include: list["clips", "refers", "sources", "simulcasts", "mentions", "description", "live_info", "channel_stats", "songs"] = None,
        lang: str = None,
        limit: int = 25,
        offset: int = 0,
        paginated: bool = False,
    ):
        params = {
            "include": ",".join(include) if include else None,
            "limit": limit,
            "offset": offset,
            "lang": lang,
            "paginated": paginated or None
        }
        data = await self.get(f"channels/{channel_id}/{video_type}", params=params)
        if data.status == 404:
            raise Exception(f"{await data.text()}")
        elif data.status == 200:
            videos = []
            total = None
            data = await data.json()
            target = data
            if paginated:
                target = data.get("items")
                total = data.get("total")
            for video in target:
                videos.append(BaseVideo.from_data(self, video))
            return videos
        return data

    async def fetch_video_topic(self, video_id):
        data = await self.get(f"videos/{video_id}/topic")
        if data.status == 404:
            raise Exception(f"{await data.text()}")
        elif data.status == 200:
            data = await data.json()
            return Topic(self,data)
        return data

    async def edit_video_topic(self, video_id, topic_id):
        body = {"videoId": video_id, "topicId": topic_id}
        data = await self.post("topics/video", body)
        if data.status == 404:
            raise Exception(f"{await data.text()}")
        elif data.status == 200:
            data = await data.json()
        return data

    async def add_video_mention(self, video_id, channel_id):
        body = {"channel_id": channel_id}
        data = await self.post(f"videos/{video_id}/mentions", body)
        if data.status == 404:
            raise Exception(f"{await data.text()}")
        elif data.status == 200:
            data = await data.json()
        return data

    async def remove_video_mentions(self, video_id, *channel_ids):
        body = {"channel_ids": list(channel_ids)}
        data = await self.delete(f"videos/{video_id}/mentions", body)
        if data.status == 404:
            raise Exception(f"{await data.text()}")
        elif data.status == 200:
            data = await data.json()
        return data

    async def fetch_video_mentions(self, video_id):
        data = await self.get(f"videos/{video_id}/mentions")
        if data.status == 404:
            raise Exception(f"{await data.text()}")
        elif data.status == 200:
            data = await data.json()
        return data

    async def fetch_orgs(self):
        data = await self.static("/statics/orgs.json")
        if data.status == 404:
            raise Exception(f"{await data.text()}")
        elif data.status == 200:
            data = await data.json()
            return [Org(o) for o in data]
        return data

    async def fill_orgs(self):
        for o in await self.fetch_orgs():
            self.insert_org(o)

    async def fetch_topics(self):
        data = await self.static("/statics/topics.json")
        if data.status == 404:
            raise Exception(f"{await data.text()}")
        elif data.status == 200:
            data = await data.json()
            return [Topic(self, t) for t in data]
        return data

    async def fill_topics(self):
        for t in await self.fetch_topics():
            self.insert_topic(t)

    async def add_placeholder(
        self,
        channel_id: str,
        name: str,
        jp_name: str,
        link: str,
        thumbnail: str,
        duration: int,
        start_time: datetime,
        credits: PlaceholderCredits,
        type: ["scheduled-yt-stream", "external-stream", "event"] = "scheduled-yt-stream",
        certainty: ["likely", "certain"] = "certain",
        id: str = None
    ):
        if not self.api_key:
            raise ValueError("No API key provided.")
        body = {
            "channel_id": channel_id,
            "title": {
                "name": name,
                "jp_name": jp_name,
                "link": link,
                "thumbnail": thumbnail,
                "placeholderType": type,
                "certainty": certainty,
                "credits": credits.to_dict()
            },
            "liveTime": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration": duration
        }
        if id:
            body["id"] = id
        data = await self.post("videos/placeholder", body)
        if data.status == 200:
            base = await data.json()
            try:
                vdata = base[0]
                return BaseVideo.from_data(self, vdata)
            except:
                vdata = base.get("placeholder")
                if vdata and (msg := base.get("error")):
                    vid = BaseVideo.from_data(self, vdata)
                    err = ValueError(msg + f" (id {vid.id})")
                    err.placeholder = vid
                    raise err
        return data

    async def delete_placeholder(self, id):
        return await self.delete(f'videos/placeholder/{id}')
            
    async def notice(self, id):
        body = {"url": id}
        data = await self.post("external/notice", body, raw=True)
        try:
            return data.get('state', data)
        except:
            return data

class BaseVideo:

    @staticmethod
    def from_data(client, data: dict):
        if (j := data.get("type")) == "stream":
            return Stream(client, data)
        elif j == "clip":
            return Clip(client, data)
        elif j == "placeholder":
            return Placeholder(client, data)
        else:
            raise ValueError(f"Invalid Video Type {j}")

    @staticmethod
    def from_youtube(client, model):
        data = {

        }
        return None

    def __init__(self, client: HolodexClient, data: dict):
        self.client = client
        self._data = data
        self.id: str = data["id"]
        if not len(self.id) == 11:
            raise ValueError("Invalid Video ID")
        self.title: str = data.get("title")
        self.type: ["stream", "clip", "placeholder"] = data.get("type")
        self.published_at: datetime | None = parse_time(data.get("published_at"))
        self.available_at: datetime = parse_time(data.get("available_at"))
        self.duration: int = parse_int(data.get("duration"))
        self.status: ["new", "upcoming", "live", "past", "missing"] = data.get("status")
        self.start_scheduled: datetime | None = parse_time(data.get("start_scheduled"))
        self.start_actual: datetime | None = parse_time(data.get("start_actual"))
        self.end_actual: datetime | None = parse_time(data.get("end_actual"))
        self.description: str = data.get("description")
        self.channel_id: str = data.get("channel_id")
        self.recommendations: list[BaseVideo] = [BaseVideo.from_data(client, video) for video in data.get("recommendations", [])]
        self.refers: list[BaseVideo] = [BaseVideo.from_data(client, video) for video in data.get("refers", [])]
        self.mentions: list[Vtuber] = [self.client.generate_channel(vtuber) for vtuber in data.get("mentions", [])]
        self.lang: str = data.get("lang")

        self.channel: Vtuber | Clipper = self.client.generate_channel(data.get("channel"))

    def __str__(self):
        return self.title

    def __repr__(self):
        return f'<BaseVideo id={self.id} type={self.type} channel={self.channel}>'

    def __eq__(self, other):
        if not isinstance(other, BaseVideo):
            return False
        return self.id == other.id

    def __ne__(self, other):
        if not isinstance(other, BaseVideo):
            return True
        return self.id != other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def thumbnail(self):
        return f"https://i.ytimg.com/vi/{self.id}/hqdefault.jpg"

    @property
    def max_thumbnail(self):
        return f"https://i.ytimg.com/vi/{self.id}/maxresdefault.jpg"

    @property
    def url(self):
        return f'https://holodex.net/watch/{self.id}'

    @property
    def hyperlink(self):
        return f'[{ed(str(self))}]({self.url})'

    @property
    def yt_url(self):
        return f'https://youtube.com/watch?v={self.id}'

    @property
    def yt_hyperlink(self):
        return f'[{ed(str(self))}]({self.yt_url})'

    async def fetch_mentions(self):
        r = await self.client.fetch_video_mentions(self.id)
        if isinstance(r, list):
            return [self.client.generate_channel(c) for c in r]

    async def add_mention(self, channel):
        if not isinstance(channel, BaseChannel):
            raise TypeError(f"Channel {c} must be of type BaseChannel, not {type(c)}.")
        return await self.client.add_video_mention(self.id, channel.id)

    async def remove_mentions(self, *channels):
        chs = []
        for c in channels:
            if not isinstance(c, BaseChannel):
                raise TypeError(f"Channel {c} must be of type BaseChannel, not {type(c)}.")
            chs.append(c.id)
        return await self.client.remove_video_mentions(self.id, *chs)

    async def remove_mention(self, channel):
        return await self.remove_mentions(self.id, channel)

    async def is_members(self):
        data = await self.client.static("https://youtube.com/watch?v=" + self.id, base=False)
        # if data.status != 200: raise
        page = await data.text()
        soup = BeautifulSoup(page, "html.parser")
        j = soup.find_all("script")
        k = (list(filter(lambda x: "playabilityStatus" in str(x), j)))
        pattern = re.compile(r"var ytInitialPlayerResponse = (.*?);$", re.MULTILINE | re.DOTALL)
        var = pattern.search(k[0].text)
        raw = var.group(1)
        final = loads(raw)
        reason = final.get("playabilityStatus", {}).get("reason", "")
        return "get access to members-only content" in reason

class Streamable(BaseVideo):

    def __init__(self, client: HolodexClient, data: dict):
        super().__init__(client, data)
        self.topic_id: str = data.get("topic_id")
        self.topic = self.client.gen_topic(self.topic_id)

    def __repr__(self):
        return f'<Streamable id={self.id} topic={self.topic} channel={self.channel}>'

    async def fetch_topic(self):
        return await self.client.fetch_video_topic(self.id)

    async def set_topic(self, topic: str | None):
        return await self.client.edit_video_topic(self.id, topic)

class Stream(Streamable):

    def __init__(self, client: HolodexClient, data: dict):
        super().__init__(client, data)
        self.clips: list[Clip] = [Clip(self.client, clip) for clip in data.get("clips", [])]
        self.simulcasts: list[Stream] = [Stream(self.client, simulcast) for simulcast in data.get("simulcasts", [])]
        self.song_count: int = parse_int(data.get("song_count")) or parse_int(data.get("songcount"))
        self.songs: list = data.get("songs")
        self.live_viewers: int | None = parse_int(data.get("live_viewers"))
        self.comments: list[Comment] = [Comment(comment) for comment in data.get("comments", [])]
        if not self.channel and data.get("channel"):
            self.channel = self.client.generate_channel(data["channel"])

    def __repr__(self):
        return f'<Stream id={self.id} topic={self.topic} channel={self.channel}>'

class Clip(BaseVideo):

    def __init__(self, client: HolodexClient, data: dict):
        super().__init__(client, data)
        self.sources: list[Stream] = [Stream(client, source) for source in data.get("sources", [])]
        self.same_source_clips: list[Clip] = [Clip.from_data(client, clip) for clip in data.get("same_source_clips", [])]

    def __repr__(self):
        return f'<Clip id={self.id} channel={self.channel}> sources={self.sources}>'

class Placeholder(Streamable):

    def __init__(self, client: HolodexClient, data: dict):
        super().__init__(client, data)
        self._thumbnail = data.get("thumbnail")
        self.jp_name = data.get("jp_name")
        self.link = data.get("link") or f'https://holodex.net/watch/{self.id}'
        self.placeholder_type = data.get("placeholderType")
        self.certainty = data.get("certainty")
        self.credits = PlaceholderCredits.from_data(data.get("credits"))
        if not self.channel and data.get("channel"):
            self.channel = self.client.generate_channel(data["channel"])

    @property
    def thumbnail(self):
        return self._thumbnail

    @property
    def url(self):
        return self.link

    @property
    def name(self):
        return self.title

    @property
    def yt_url(self):
        return None

    def __repr__(self):
        return f'<Placeholder id={self.id} type={self.type} channel={self.channel}>'

    async def edit(
        self,
        title: str = None,
        jp_name: str = None,
        link: str = None,
        thumbnail: str = None,
        duration: int = None,
        start_time: datetime = None,
        credits: PlaceholderCredits = None,
        type: ["scheduled-yt-stream", "external-stream", "event"] = None,
        certainty: ["likely", "certain"] = None
    ):
        return await self.client.add_placeholder(
            self.channel.id,
            title or self.title,
            jp_name or self.jp_name,
            link or self.link,
            thumbnail or self.thumbnail,
            duration or self.duration,
            start_time or self.start_scheduled or self.available_at,
            credits or self.credits,
            type=type or self.placeholder_type,
            certainty=certainty or self.certainty,
            id=self.id
        )

    async def delete(self):
        return await self.client.delete_placeholder(self.id)


class BaseChannel:

    @staticmethod
    def from_data(client, data: dict):
        if (j := data.get("type")) == "vtuber":
            return client.cls(client, data)
        elif j == "subber":
            return Clipper(client, data)
        else:
            raise ValueError(f"Invalid Channel Type {j}")

    def __init__(self, client: HolodexClient, data: dict):
        self.client = client
        self.id: str = data["id"]
        if not self.id.startswith("UC"):
            raise ValueError("Invalid channel ID")
        self.name: str = data.get("name")
        self.type: ["vtuber", "subber"] = data.get("type")
        self.description: str = data.get("description", "")
        self._photo: str | None = data.get("photo")
        self.thumbnail: str | None = data.get("thumbnail")
        self.banner: str | None = data.get("banner")
        if self.banner and "=w" not in self.banner:
            self.banner += "=w10000"
        self.view_count: int | None = parse_int(data.get("view_count"))
        self.subscriber_count: int | None = parse_int(data.get("subscriber_count"))
        self.video_count: int | None = parse_int(data.get("video_count"))
        self.published_at: datetime | None = parse_time(data.get("published_at"))
        self.updated_at: datetime | None = parse_time(data.get("updated_at"))
        self.crawled_at: datetime | None = parse_time(data.get("crawled_at"))
        self.created_at: datetime | None = parse_time(data.get("created_at"))
        self.yt_uploads_id: str | None = data.get("yt_uploads_id")
        self.twitter: str | None = data.get("twitter")
        if self.twitter:
            self.twitter = self.twitter.replace("@", "")
        self.inactive: bool = data.get("inactive", False)
        self.yt_handle: list = data.get("yt_handle", [])
        self.yt_name_history: list = data.get("yt_name_history", [])

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'<BaseChannel id={self.id} name={self.name!r}'

    def __eq__(self, other):
        if not isinstance(other, BaseChannel):
            return False
        return self.id == other.id

    def __ne__(self, other):
        if not isinstance(other, BaseChannel):
            return True
        return self.id != other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def photo(self):
        return f'https://holodex.net/statics/channelimg/{self.id}.png'

    @property
    def url(self):
        return f'https://holodex.net/channel/{self.id}'

    @property
    def hyperlink(self):
        return f'[{ed(str(self))}]({self.url})'

    @property
    def twitter_url(self):
        if self.twitter:
            return f'https://twitter.com/{self.twitter}'
        return None

    @property
    def twitter_hyperlink(self):
        if self.twitter:
            return f'[@{ed(self.twitter)}]({self.twitter_url})'
        return None

    @property
    def yt_url(self):
        return f'https://youtube.com/channel/{self.id}'

    @property
    def yt_hyperlink(self):
        return f'[{ed(self.name)}]({self.yt_url})'

    @property
    def handle(self):
        return self.yt_handle[0] if self.yt_handle else None

    @property
    def handle_url(self):
        if self.handle:
            return f'https://www.youtube.com/{self.handle}'
        return None

    @property
    def handle_hyperlink(self):
        if self.handle:
            return f'[{ed(self.handle)}]({self.handle_url})'
        return None

    # @handle.setter
    # def handle(self, value):
    #     if value not in self.yt_handle and value:
    #         self.yt_handle = [value] + self.yt_handle
    #     elif value is None:
    #         raise ValueError("Cannot unset handle.")

    @property
    def info(self):
        return f'{self.hyperlink} - ID {self.id}'

    def format_created(self, style: str | list = "f", separator: str = "", timestamp=None) -> str | None:
        timestamp = timestamp or self.created
        if not timestamp:
            return None
        if isinstance(style, str):
            style = [style]
        return separator.join([fdt(timestamp, style=s) for s in style])

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "photo": self.photo,
            "yt_handle": self.yt_handle,
            "twitter": self.twitter,
            "twitter_id": self.twitter_id,
            "inactive": self.inactive,
            "type": self.type
        }

    async def fetch_videos(self, **kwargs):
        return await self.client.fetch_channel_videos(self.id, "videos", **kwargs)

class Vtuber(BaseChannel):

    def __init__(self, client: HolodexClient, data: dict):
        super().__init__(client, data)
        if self.type != "vtuber":
            raise TypeError("Channel is not a vtuber.")
        self.english_name: str | None = data.get("english_name")
        self.org: Org | None = data.get("org")
        self.suborg: str | None = data.get("suborg")
        self.group: str | None = data.get("group") or (data.get("suborg", "") or "")[2:]
        self.clip_count: int | None = parse_int(data.get("clip_count"))
        self.comments_crawled_at: datetime | None = parse_time(data.get("comments_crawled_at"))
        self.top_topics: list = [Topic.from_str(client, t) for t in data.get("top_topics", []) or []]
        self.twitch: str | None = data.get("twitch")
        self.twitter_id: int | None = parse_int(data.get("twitter_id"))

        self.transform()

    def transform(self):
        if isinstance(self.org, str):
            self.org = self.client.get_org(self.org) or Org(self.client, self.org)

    def __str__(self):
        return self.english_name or self.name

    def __repr__(self):
        return f'<Vtuber id={self.id} name={self.name!r} english_name={self.english_name!r}>'

    @property
    def full_org(self):
        return self.org.name + f'{f" {self.group}" if self.group else ""}'

    @property
    def twitch_url(self):
        if self.twitch:
            return f'https://twitch.tv/{self.twitch}'
        return None

    @property
    def twitch_hyperlink(self):
        if self.twitch:
            return f'[{ed(self.twitch)}]({self.twitch_url})'
        return

    def convert_dict(self):
        return {
            "name": self.name,
            "display": self.english_name,
            "org": self.org.name if self.org else None,
            "group": self.group,
            "image": self.photo,
            "brand": self.handle,
            "twitter": self.twitter,
            "twitter_id": self.twitter_id,
            "twitch": self.twitch,
            "inactive": self.inactive,
        }

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "english_name": self.english_name,
            "org": self.org,
            "group": self.group,
            "photo": self.photo,
            "yt_handle": self.yt_handle,
            "twitter": self.twitter,
            "twitch": self.twitch,
            "inactive": self.inactive,
            "type": self.type
        }

    async def fetch_clips(self, **kwargs):
        return await self.client.fetch_channel_videos(self.id, "clips", **kwargs)

    async def fetch_collabs(self, **kwargs):
        return await self.client.fetch_channel_videos(self.id, "collabs", **kwargs)

    async def add_placeholder(self, *args, **kwargs):
        return await self.client.add_placeholder(self.id, *args, **kwargs)

class Clipper(BaseChannel):

    def __init__(self, client: HolodexClient, data: dict):
        super().__init__(client, data)
        if self.type != "subber":
            raise TypeError("Channel is not a clipper.")
        self.lang: str | None = data.get("lang")

    def __repr__(self):
        return f'<Clipper id={self.id} name={self.name!r}>'

class Org:

    def __init__(self, client, name):
        self.client = client
        self.name = name

    def __repr__(self):
        return f'<Org name={self.name!r} members={len(self.members)}>'

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if not isinstance(other, Org):
            return False
        return self.name == other.name

    @property
    def members(self):
        return self.client.find_channels(org=self)

    @property
    def groups(self):
        allGroups = [v.group for v in self.members if v.group]
        return list(set(allGroups))

    async def fetch_videos(self, **kwargs):
        return await self.client.fetch_videos(org=self.name, **kwargs)

class Topic:

    @staticmethod
    def from_str(client, name):
        return Topic(client, {"id": name})

    def __init__(self, client, data):
        self.client = client
        self.id: str = data["id"]
        self.count: int = data.get("count", None)
        self.approver_id: int = data.get("topic_approver_id", data.get("approver_id", None))

    def __repr__(self):
        return f'<Topic id={self.id!r}> count={self.count}>'

    def __str__(self):
        return self.id

    def __eq__(self, other):
        if not isinstance(other, Topic):
            return False
        return self.id == other.id

    async def fetch_videos(self, **kwargs):
        return await self.client.fetch_videos(topic=self.id, **kwargs)
