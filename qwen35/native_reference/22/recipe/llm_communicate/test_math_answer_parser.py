"""Whole-answer regressions: roots, fractions, sets, functions, relations and numeric false positives."""
import unittest
from unittest.mock import patch

from math_answer_parser import normalize_math_answer, score_final_answer, SCORING_VERSION
import qwen35_bridge_opd_score_worker as worker


class Tests(unittest.TestCase):
    def test_equivalent_entire_expressions(self):
        cases=[('103','103'),('0','0'),('-3','-3'),
            (r'8\sqrt{10}',r'8\sqrt{10}'),
            (r'\frac{9\sqrt{23}}{23}',r'9\sqrt{23}/23'),
            (r'1-\frac{2}{\pi}',r'1-\frac{2}{\pi}'),
            (r'\frac{1}{2}','0.5'),
            (r'5\pi+6\sqrt{3}',r'6\sqrt{3}+5\pi'),
            ('8,9','9,8'),(r'$2^{u-2}$','2^{u-2}'),
            (r'$0<k\le\frac{4}{9}$','0 < k <= 4/9'),
            (r'$f(x)=x^2-x$',r'f(x)=x(x-1)'),
            (r'$(a+b)^2$',r'a^2+2ab+b^2'),
            (r'$\sqrt{2}-1$',r'\boxed{\sqrt{2}-1}'),
            (r'$p_1^{p_1^{e_1}-1}\cdots p_k^{p_k^{e_k}-1}$',
             r'$p_1^{p_1^{e_1}-1}\cdots p_k^{p_k^{e_k}-1}$')]
        for gold,answer in cases:
            with self.subTest(gold=gold,answer=answer):
                result=score_final_answer(gold,answer)
                self.assertEqual(result['acc'],1)
                self.assertFalse(result['scoring_timed_out'])
                self.assertEqual(result['scoring_version'],SCORING_VERSION)

    def test_reject_partial_numeric_matches(self):
        for gold,answer in [('4','4, 5, 6'),('4','4, 6'),('2',r'n\ge 2'),
                            ('3','3 if n=3, 2 if n>=4'),(r'8\sqrt{10}','8'),('7','17')]:
            with self.subTest(gold=gold,answer=answer):
                self.assertEqual(score_final_answer(gold,answer)['acc'],0)

    def test_full_text_exact_fallback_is_labeled(self):
        value='1 if triangle is isoceles, otherwise 2'
        result=score_final_answer(value,value)
        self.assertEqual(result['acc'],1)
        self.assertTrue(result['gold_parse_failed'])
        self.assertEqual(result['verification_mode'],'whole_answer_exact_only')
        self.assertEqual(score_final_answer(value,'2')['acc'],0)

    def test_normalization_preserves_multiple_math_segments(self):
        value=r'$(x,y)$ for $x>1$'
        self.assertEqual(normalize_math_answer(value),value)
        self.assertEqual(normalize_math_answer(r'Answer: $\frac{1}{2}$.'),r'\frac{1}{2}')

    def test_last_answer_only_and_length_gate(self):
        row=dict(response=r'Answer: 99'+'\nIntermediate 7\n'+r'Answer: 8\sqrt{10}',
                 finish_reason='stop',gold=r'8\sqrt{10}',gold_missing=False,acc=0,score=0.)
        result=worker.add_answer_metrics(row)
        self.assertEqual(result['acc_answer'],1)
        self.assertEqual(result['acc_strict'],1)
        result=worker.add_answer_metrics(dict(row,finish_reason='length'))
        self.assertEqual(result['acc_answer'],0)
        self.assertEqual(result['acc_strict'],0)
        result=worker.add_answer_metrics(dict(row,response='Intermediate '+r'$8\sqrt{10}$'))
        self.assertEqual(result['acc_answer'],0)

    def test_training_format_uses_same_parser(self):
        def legacy(row):
            return dict(row,acc=0,score=0.,pred=None,gold_missing=False,scoring_timed_out=False)
        row=dict(gold=r'8\sqrt{10}',response=r'Answer: 8\sqrt{10}',finish_reason='stop')
        with patch.object(worker,'score',side_effect=legacy):
            result=worker.score_with_format(row)
            self.assertEqual(result['acc'],1)
            self.assertEqual(result['acc_unpenalized'],0)
            self.assertEqual(worker.score_with_format(dict(row,response=r'Answer: 8\sqrt{10}'+'\nExtra'))['acc'],0)

    def test_timeout_is_not_silently_accepted(self):
        with patch('math_verify.parse',side_effect=TimeoutError):
            result=score_final_answer('7','7')
        self.assertTrue(result['scoring_timed_out'])
        with patch.object(worker,'score_final_answer',return_value=result):
            self.assertIsNone(worker._verify_line(dict(gold='7'),'Answer: 7'))

    def test_actual_math_verify_comparison_timeout(self):
        from math_verify.errors import TimeoutException
        with patch('math_verify.verify',side_effect=TimeoutException('Timeout during comparison')):
            result=score_final_answer('7','7')
        self.assertTrue(result['scoring_timed_out'])
        self.assertEqual(result['timeout_error'],'TimeoutException')
        with patch.object(worker,'score_final_answer',side_effect=[result,dict(result,scoring_timed_out=False,acc=1)]) as verify:
            self.assertEqual(worker._verify_line(dict(gold='7'),'Answer: 7'),1)
        self.assertEqual([c.kwargs['timeout_seconds'] for c in verify.call_args_list],[worker.TIMEOUT,2*worker.TIMEOUT])


if __name__=='__main__':
    unittest.main()
