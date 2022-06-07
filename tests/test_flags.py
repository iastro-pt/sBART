from SBART.utils.status_codes import Flag


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