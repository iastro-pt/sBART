from SBART.utils.status_codes import Flag, Status
from SBART.utils import status_codes


def test_Model_component_init():
    flag_1 = Flag(name="teste",
                  value="A",
                  description="teste flag",
                  fatal_flag=True,
                  is_warning=False,
                  is_good_flag=False
                  )

    assert flag_1.is_fatal
    assert not flag_1.is_warning
    assert not flag_1.is_good_flag

    flag_1 = Flag(name="teste",
                  value="A",
                  description="teste flag",
                  fatal_flag=False,
                  is_warning=False,
                  is_good_flag=True
                  )
    assert flag_1.is_good_flag


def test_flag_equality():
    flag_list = []
    for index in range(2):
        f = Flag(name="teste",
             value="A",
             description="teste flag",
             fatal_flag=True,
             is_warning=False,
             is_good_flag=False
             )

        flag_list.append(f)

    assert flag_list[0] == flag_list[1]

    updated_flag_1 = flag_list[1]("Extra information")

    assert flag_list[1] != updated_flag_1


def test_flag_storage():
    flag_1 = Flag(name="teste",
             value="A",
             description="teste flag",
             fatal_flag=True,
             is_warning=False,
             is_good_flag=False
             )
    flag_1_json = flag_1.to_json()

    flag1_loaded = Flag.create_from_json(flag_1_json)

    assert flag_1 == flag1_loaded


def test_Status():
    assert not Status(assume_valid=False).is_valid
    assert Status(assume_valid=True).is_valid

    stat = Status(assume_valid=True)
    assert stat.has_flag(status_codes.VALID)
    assert stat.number_warnings == 0

    warningflag = status_codes.KW_WARNING("KW foo meets the bad value")
    stat.store_warning(warningflag)
    assert stat.check_if_warning_exists(warningflag)
    warningflag = status_codes.KW_WARNING("KW bar meets the bad value")
    assert not stat.check_if_warning_exists(warningflag)
    assert stat.is_valid
