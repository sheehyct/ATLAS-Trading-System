Concurrent Requests
This is a guide to making concurrent requests for this REST API. Depending on your subscription tier, you are allowed a different number of outstanding requests. Your requests are NOT rate-limited in any way, but the number of outstanding requests are limited.

The table below describes the number of outstanding requests by subscription tier.

Subscription Tier	Outstanding Requests
FREE	1
VALUE	2
STANDARD	4
PRO	8
You can make more requests than the number listed in the table above; however, they will be queued. The default queue size is 16, but can be configured up to 128. If the queue grows beyond its specified size, you may receive a 429 response from the server.

What are Concurrent Requests?
Making multiple requests at the same time instead of waiting for a response for each request sequentially. This will improve performance since a large part of a request's overhead can be the round trip between your computer and our servers. If it takes 100ms to get back a response from Theta Data due to your network, then 100ms per request is being wasted waiting for a response. If you make multiple requests at the same time, but staying within your limits, this overhead is eliminated.

Programming Considerations
You will need to be familiar with concurrency and multithreadding in the programming language you use. The best way to ensure you are not making more requests than your allowed number, is to leverage a semaphore.