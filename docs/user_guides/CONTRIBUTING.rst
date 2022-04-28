============================
Contributing Guidelines
============================

Installing dependencies
===========================

If using poetry:

.. code-block:: console

    poetry install

Otherwise:

.. code-block:: console

    pip install -r dev_requirements.txt

Changelogs
============

We are currently using `towncrier <https://github.com/twisted/towncrier>`_ as the generator of the
changelog file. Small changes (e.g. typos / code style) don't need to be a "news fragment", but all others
**must have** an `issue <https://github.com/iastro-pt/sBART/issues>`_ associated with it and
**must be** included into the news.


Quick overview of towncrier
--------------------------------

In short, this creates "News Fragments" from custom files that live (until the changelog
file is constructed) inside the "newsfragments" folder.

The files must be named with an issue number and a `news type <https://towncrier.readthedocs.io/en/latest/index.html?highlight=feature#news-fragments>`_.
Inside, we must have a short, non-technical, description of the change

.. note::

    As an example, we can look at the news from `issue 1 <https://github.com/iastro-pt/sBART/blob/6e9caea093619e2cb8d7027ef41edfbe6247459e/newsfragments/1.feature>`_.
