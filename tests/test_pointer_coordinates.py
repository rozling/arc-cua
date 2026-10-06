"""Deterministic pointer contract regressions; no native events or sleeping."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from arc_cua.backends import macos_app, macos_background as bg
from arc_cua.driver import Driver, WindowTarget
from arc_cua.errors import UnsupportedDesktopAction
from arc_cua.models import Bounds


@pytest.fixture
def pointer_api(monkeypatch):
    events, lookups = [], []
    rows = []

    def mouse(source, kind, point, button):
        event = SimpleNamespace(kind=kind, global_point=point, local_point=None)
        events.append(event)
        return event

    def scroll(*args):
        return mouse(None, 6, None, None)

    def window_info(option, window):
        lookups.append((option, window))
        return rows

    q = SimpleNamespace(
        kCGWindowListOptionIncludingWindow=8, kCGWindowNumber='number',
        kCGWindowOwnerPID='pid', kCGWindowBounds='bounds',
        CGWindowListCopyWindowInfo=window_info,
        kCGEventSourceStateHIDSystemState=0, CGEventSourceCreate=lambda _: object(),
        kCGMouseButtonLeft=0, kCGMouseButtonRight=1,
        kCGEventMouseMoved=0, kCGEventLeftMouseDown=1, kCGEventLeftMouseUp=2,
        kCGEventLeftMouseDragged=3, kCGEventRightMouseDown=4, kCGEventRightMouseUp=5,
        CGEventCreateMouseEvent=mouse, CGEventSetFlags=lambda *args: None,
        kCGScrollEventUnitPixel=0, CGEventCreateScrollWheelEvent=scroll,
        CGEventSetLocation=lambda event, point: setattr(event, 'global_point', point),
    )
    sl = SimpleNamespace(
        SLEventSetIntegerValueField=lambda *args: None,
        CGEventSetWindowLocation=lambda event, x, y: setattr(event, 'local_point', (x, y)),
    )
    monkeypatch.setattr(bg, '_quartz', lambda: q)
    monkeypatch.setattr(bg, '_sl', lambda: sl)
    monkeypatch.setattr(bg, '_pointer', lambda event: event)
    monkeypatch.setattr(bg, '_post', lambda *args: None)
    monkeypatch.setattr(bg, 'ensure_available', lambda: None)
    monkeypatch.setattr(bg, 'borrowed_focus', lambda *args: nullcontext())
    monkeypatch.setattr(bg.time, 'sleep', lambda _: None)
    return events, lookups, rows


@pytest.mark.parametrize('origin', [(60, 80), (-1440, 100)])
@pytest.mark.parametrize('operation', ['click', 'double', 'right', 'drag', 'scroll'])
def test_public_driver_pointer_spaces(monkeypatch, pointer_api, origin, operation):
    events, lookups, rows = pointer_api
    rows.append({'number': 9, 'pid': 42, 'bounds': {'X': origin[0], 'Y': origin[1]}})
    window = SimpleNamespace(window_id=9, bounds=Bounds(*origin, 500, 500))
    app = object.__new__(macos_app.MacOSApp)
    app.pid, app.ax = 42, object()
    app._input_window = lambda *args: window
    app.input_scope = nullcontext
    monkeypatch.setattr(macos_app, 'scroll_area_by_bars', lambda *args, **kwargs: False)
    driver = object.__new__(Driver)
    target = WindowTarget(42, 9)
    driver._input_target = lambda *args: (target, None)
    driver._window = lambda _: window
    driver._app = lambda _: SimpleNamespace(app=app)
    driver._raw = lambda target, send, *args: send()
    if operation == 'drag':
        driver.drag(target, [(120, 226), (144, 250)])
    elif operation == 'scroll':
        driver.scroll_at(target, 120, 226, dy=10)
    else:
        driver.click_at(target, 120, 226, count=2 if operation == 'double' else 1,
                        button='right' if operation == 'right' else 'left')
    assert lookups == [(8, 9)]  # One lookup for the whole event stream.
    for event in events:
        assert event.local_point == (event.global_point[0] - origin[0], event.global_point[1] - origin[1])
    assert events[0].global_point == (origin[0] + 120, origin[1] + 226)
    assert events[0].local_point == (120, 226)
    if operation in ('click', 'double'):
        assert [event.local_point for event in events[1:3]] == [(-1, -1), (-1, -1)]
        assert events[1].global_point == (origin[0] - 1, origin[1] - 1)
    if operation == 'drag':
        assert events[-1].local_point == (144, 250)
        assert any(event.kind == 3 for event in events)


@pytest.mark.parametrize('bounds', [None, {}, {'X': float('nan'), 'Y': 80}])
@pytest.mark.parametrize('operation', ['click', 'drag_path', 'scroll'])
def test_missing_origin_refuses_before_events(pointer_api, bounds, operation):
    events, _, rows = pointer_api
    if bounds is not None:
        rows.append({'number': 9, 'pid': 42, 'bounds': bounds})
    args = [(180, 306), (200, 330)] if operation == 'drag_path' else (180, 306)
    with pytest.raises(UnsupportedDesktopAction, match='window origin'):
        getattr(bg, operation)(42, 9, args)
    assert events == []
