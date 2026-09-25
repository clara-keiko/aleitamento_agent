from app import guardrails
from app.guardrails import EDUCATIONAL_OK, EMERGENCY_NOW, REFER_MEDICAL_CARE, classify_risk


class TestEmergencia:
    def test_sem_acento_ainda_dispara(self):
        # O bug clássico: mãe em pânico não digita acento.
        assert classify_risk("socorro meu bebe nao respira").level == EMERGENCY_NOW

    def test_com_acento(self):
        assert classify_risk("meu bebê não está respirando").level == EMERGENCY_NOW

    def test_convulsao(self):
        assert classify_risk("ela teve uma convulsão agora").level == EMERGENCY_NOW

    def test_cianose(self):
        assert classify_risk("o bebê ficou roxo").level == EMERGENCY_NOW

    def test_emergencia_vence_pergunta_geral(self):
        # Mesmo em forma de pergunta, sinal de emergência não é educativo.
        risco = classify_risk("é normal o bebê ficar roxo e com falta de ar?")
        assert risco.level == EMERGENCY_NOW


class TestEncaminhamento:
    def test_relato_de_febre(self):
        assert classify_risk("meu bebê está com febre desde ontem").level == REFER_MEDICAL_CARE

    def test_recusa_de_mamar(self):
        assert classify_risk("ele não quer mamar faz 2 dias").level == REFER_MEDICAL_CARE

    def test_pouco_xixi(self):
        assert classify_risk("minha bebê está com pouco xixi").level == REFER_MEDICAL_CARE

    def test_frase_sem_cara_de_pergunta_encaminha(self):
        assert classify_risk("sangue nas fezes").level == REFER_MEDICAL_CARE


class TestPerguntaGeralSensivel:
    """A regressão que motivou a mudança: dúvida conceitual era engolida."""

    def test_pergunta_sobre_febre_é_respondida_com_nota(self):
        risco = classify_risk("posso amamentar se eu estiver com febre?")
        assert risco.level == EDUCATIONAL_OK
        assert risco.needs_safety_note is True

    def test_pergunta_sobre_mastite_é_respondida_com_nota(self):
        risco = classify_risk("o que é mastite e como prevenir?")
        assert risco.level == EDUCATIONAL_OK
        assert risco.needs_safety_note is True

    def test_nota_de_seguranca_menciona_profissional(self):
        assert "profissional de saúde" in guardrails.safety_note()


class TestEducativo:
    def test_duvida_comum(self):
        risco = classify_risk("qual a melhor posição para amamentar?")
        assert risco.level == EDUCATIONAL_OK
        assert risco.needs_safety_note is False

    def test_livre_demanda(self):
        assert classify_risk("preciso dar mamada de 3 em 3 horas?").level == EDUCATIONAL_OK

    def test_texto_vazio(self):
        assert classify_risk("").level == EDUCATIONAL_OK


class TestFalsosPositivos:
    def test_palavra_dentro_de_outra_nao_dispara(self):
        # "roxo" não pode casar dentro de "roxinho" num contexto de roupa.
        assert classify_risk("comprei uma roupa lilás para ela").level == EDUCATIONAL_OK

    def test_assunto_neutro(self):
        assert classify_risk("como faço para doar leite humano?").level == EDUCATIONAL_OK


class TestOptOut:
    def test_variantes(self):
        for palavra in ["sair", "SAIR", "Parar", "cancelar", "stop", "sair."]:
            assert guardrails.is_opt_out(palavra) is True

    def test_frase_normal_nao_e_opt_out(self):
        assert guardrails.is_opt_out("quero parar de dar mamadeira") is False


class TestNormalizacao:
    def test_remove_acento_e_caixa(self):
        assert guardrails.normalize("Não É  Fácil") == "nao e facil"

    def test_none_seguro(self):
        assert guardrails.normalize("") == ""
