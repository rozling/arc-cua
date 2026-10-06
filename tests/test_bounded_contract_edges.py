"""Edge cases for the public optional Subtask contract and proposal boundary."""
from dataclasses import replace

import pytest

from arc_cua import ActionKind, DesktopElement, DesktopSnapshot, TerminalKind
from test_bounded_decisions import _contract, _run, _snapshot, _task


@pytest.mark.parametrize('change', [
    lambda c: c['bindings']['search'].update(name=' '),
    lambda c: c['bindings']['search'].update(name='x' * 4097),
    lambda c: c['bindings']['search'].update(source=None),
    lambda c: c['intents'][0].update(input_key='not-supplied'),
    lambda c: c['intents'][0].update(kind='raw_drag'),
    lambda c: c['intents'][0].update(extra=1),
    lambda c: c['verification'][0].update(criterion_index=True),
    lambda c: c['verification'][0].update(criterion_index=99),
    lambda c: c['verification'][0]['all'][0].update(field='bounds'),
    lambda c: c['verification'][0]['all'][0].update(value=None),
    lambda c: c['verification'][0]['all'][0].update(input_key=None),
    lambda c: c['verification'][0]['all'][0].update(value=float('nan')),
    lambda c: c['verification'][0]['all'][0].update(value=0),
    lambda c: c['verification'][0].update(all=[]),
    lambda c: c.update(intents=[]),
])
def test_invalid_contract_rejects_at_subtask_boundary(change):
    contract = _contract()
    change(contract)
    with pytest.raises(ValueError):
        _task(decision_contract=contract)


def test_replacing_typed_subtask_revalidates_new_inputs_and_criteria():
    task = _task()
    assert replace(task, goal='Same bindings, new wording').decision_contract == task.decision_contract
    with pytest.raises(ValueError):
        replace(task, inputs={})
    with pytest.raises(ValueError):
        replace(task, verification=('Only one criterion',))


@pytest.mark.parametrize('element_change', [
    {'actions': ()},
    {'role': 'Button', 'metadata': {}},
    {'visible': False},
    {'enabled': False},
])
def test_unsupported_intent_target_does_not_reach_provider(element_change):
    snapshot = _snapshot()
    rows = (snapshot.elements[0], replace(snapshot.elements[1], **element_change), snapshot.elements[2])
    decision, bodies, calls = _run(_task(), replace(snapshot, elements=rows))
    assert decision.terminal == TerminalKind.NEEDS_AGENT
    assert calls == 0 and bodies == []


@pytest.mark.parametrize('current,desired', [(None, False), (2, False), (False, 2), ('0', False), (False, 'off')])
def test_toggle_requires_known_binary_values(current, desired):
    task = _task(inputs={'query': 'needle', 'filter_off': desired})
    decision, _, calls = _run(task, _snapshot(filter_value=current))
    assert decision.terminal == TerminalKind.NEEDS_AGENT and calls == 0


def test_each_multi_field_proposal_materializes_its_own_literal():
    from arc_cua.validation import materialize_action
    contract = _contract()
    contract['bindings']['second'] = {'role': 'TextField', 'name': 'Second', 'parent': 'browser'}
    contract['intents'].append({'id': 'set_second', 'kind': 'set_text', 'binding': 'second', 'input_key': 'second'})
    task = _task(inputs={'query': 'first literal', 'filter_off': False, 'second': 'second literal'},
                 decision_contract=contract)
    snapshot = _snapshot()
    snapshot = replace(snapshot, elements=(*snapshot.elements,
        DesktopElement('second', 'TextField', 'Second', value='', parent_id='browser', actions=(ActionKind.TYPE_TEXT,))))
    for target_id, expected in (('search', 'first literal'), ('second', 'second literal')):
        def select(body):
            return next(key for key, proposal in body['state']['proposals'].items() if proposal['target_id'] == target_id)
        decision, _, _ = _run(task, snapshot, selector=select)
        action = materialize_action(decision, snapshot, task)
        assert (action.target_id, action.value) == (target_id, expected)


def test_cycle_in_observed_tree_is_a_no_spend_handoff():
    snapshot = _snapshot()
    snapshot = replace(snapshot, elements=(replace(snapshot.elements[0], parent_id='search'), *snapshot.elements[1:]))
    decision, _, calls = _run(_task(), snapshot)
    assert decision.reason == 'cyclic_observed_ancestry' and calls == 0


def test_explicit_hidden_view_satisfies_but_missing_view_does_not():
    contract = _contract()
    contract['bindings']['view'] = {'role': 'Group', 'name': 'Filter View', 'parent': 'browser'}
    contract['verification'][1]['all'].append({'binding': 'view', 'field': 'visible', 'value': False})
    task = _task(decision_contract=contract)
    snapshot = _snapshot(search_value='needle', filter_value=False)
    decision, _, calls = _run(task, snapshot)
    assert decision.terminal == TerminalKind.NEEDS_AGENT and calls == 0
    assert decision.raw['decision_contract']['evidence'][1]['status'] == 'UNKNOWN'
    hidden = DesktopElement('view', 'Group', 'Filter View', parent_id='browser', visible=False)
    decision, bodies, calls = _run(task, replace(snapshot, elements=(*snapshot.elements, hidden)),
                                  selector=lambda body: 'SUBTASK_COMPLETE')
    assert decision.terminal == TerminalKind.SUBTASK_COMPLETE and calls == 1
    assert bodies[0]['state']['bindings']['view']['visible'] is False


def test_unrelated_ui_omission_does_not_veto_covered_completion():
    snapshot = _snapshot(search_value='needle', filter_value=False)
    noise = tuple(DesktopElement(f'n{i}', 'Button', f'Unrelated {i}') for i in range(600))
    decision, bodies, _ = _run(_task(), replace(snapshot, elements=(*snapshot.elements, *noise)),
                               selector=lambda body: 'SUBTASK_COMPLETE')
    assert decision.terminal == TerminalKind.SUBTASK_COMPLETE
    assert decision.raw['provider_packing']['dropped_elements'] == 600
    assert 'Unrelated' not in str(bodies)


def test_conflicting_intents_cannot_oscillate_even_if_first_already_satisfied():
    contract = _contract()
    contract['intents'] = [
        {'id': 'first', 'kind': 'set_text', 'binding': 'search', 'input_key': 'query'},
        {'id': 'second', 'kind': 'set_text', 'binding': 'search', 'input_key': 'other'},
    ]
    task = _task(inputs={'query': 'needle', 'other': 'different', 'filter_off': False}, decision_contract=contract)
    decision, _, calls = _run(task, _snapshot(search_value='needle'))
    assert decision.reason == 'conflicting_intents' and calls == 0


def test_same_single_scalar_not_equal_to_nonfinite_observation():
    contract = _contract()
    contract['verification'][0] = {'criterion_index': 0, 'all': [{'binding': 'browser', 'field': 'value', 'value': 1.0}]}
    task = _task(decision_contract=contract)
    snapshot = _snapshot()
    snapshot = replace(snapshot, elements=(replace(snapshot.elements[0], value=float('nan')), *snapshot.elements[1:]))
    # Nonfinite facts are not JSON representable: fail before HTTP, never claim completion.
    with pytest.raises(ValueError):
        _run(task, snapshot)
